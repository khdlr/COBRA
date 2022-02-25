import sys
import yaml
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import tqdm

import models
from data_loading import get_loader
from lib import losses, utils, logging
from lib.utils import TrainingState, prep, load_state


METRICS = dict(
    mae            = losses.mae,
    rmse           = losses.rmse,
    forward_mae    = losses.forward_mae,
    backward_mae   = losses.backward_mae,
    forward_rmse   = losses.forward_rmse,
    backward_rmse  = losses.backward_rmse,
    symmetric_mae  = losses.symmetric_mae,
    symmetric_rmse = losses.symmetric_rmse,
)


@partial(jax.jit, static_argnums=3)
def test_step(batch, state, key, net):
    imagery, mask, contour = prep(batch)

    terms, _ = net(state.params, state.buffers, key, imagery, is_training=False)

    terms = {
        **terms,
        'imagery': imagery,
        'contour': contour,
        'mask': mask,
    }

    if 'snake' not in terms:
      terms['snake'] = utils.snakify(preds, contour.shape[-2])
    if 'snake_steps' not in terms:
      terms['snake_steps'] = [terms['snake']]

    # Convert from normalized to to pixel coordinates
    scale = imagery.shape[1] / 2
    for key in ['snake', 'snake_steps', 'contour']:
      terms[key] = jax.tree_map(lambda x: scale * (1.0 + x), terms[key])

    metrics = {}
    for m in METRICS:
        metrics[m] = call_loss(METRICS[m], terms)

    return metrics, terms


if __name__ == '__main__':
    run = Path(sys.argv[1])
    assert run.exists()
    do_output = True

    config = yaml.load(open(run / 'config.yml'), Loader=yaml.SafeLoader)
    config['dataset'] = 'CALFIN'
    config['data_root'] = '../CALFIN/training/data'
    config['data_channels'] = [2]

    datasets = ['validation' , 'validation_baumhoer', 'validation_zhang']
    loaders  = {d: get_loader(4, 1, d, config, None, subtiles=False) for d in datasets}

    config['dataset'] = 'TUD'
    config['data_root'] = '../aicore/uc1/data/'
    config['data_channels'] = ['SPECTRAL/BANDS/STD_2s_B8_8b']
    loaders['TUD_test'] = get_loader(4, 1, 'test', config, subtiles=False)

    for sample_batch in loaders[datasets[0]]:
        break
    img, *_ = prep(sample_batch)

    S, params, buffers = models.get_model(config, img)
    state = utils.load_state(run / 'latest.pkl')
    net = S.apply

    all_metrics = {}
    for dataset, loader in loaders.items():
        test_key = jax.random.PRNGKey(27)
        test_metrics = {}
        for batch in tqdm(loader, desc=dataset):
            test_key, subkey = jax.random.split(test_key)
            metrics, output = test_step(batch, state, subkey, net)

            for m in metrics:
              if m not in test_metrics: test_metrics[m] = []
              test_metrics[m].append(metrics[m])

        logging.log_metrics(test_metrics, dataset, 0, do_wandb=False)
        for m in test_metrics:
            all_metrics[f'{dataset}/{m}'] = np.mean(test_metrics[m])

    with (run / 'metrics.json').open('w') as f:
        print(all_metrics, file=f)

        # print(f'{"Metric".ljust(15)}:    mean   median      min        max')
        # for m, val in metrics.items():
        #     print(f'{m.ljust(15)}: {val.mean():7.4f}  {jnp.median(val):7.4f}  '
        #           f'{jnp.min(val):7.4f} – {jnp.max(val):8.4f}')
        # print()
        # if do_output:
        #     with open('results.csv', 'a') as f:
        #         for m, val in metrics.items():
        #             print(f'{run.stem},{config["run_id"]},{dataset},{m},'
        #                   f'{val.mean()},{jnp.median(val)},{jnp.mean(val)},{jnp.min(val)},{jnp.max(val)}',
        #                     file=f)
