import sys
import yaml
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm

import models
from data_loading import get_loader
from lib import metrics, utils
from lib.utils import TrainingState, prep, load_state


METRICS = dict(
    mae            = metrics.mae,
    rmse           = metrics.rmse,
    forward_mae    = metrics.forward_mae,
    backward_mae   = metrics.backward_mae,
    forward_rmse   = metrics.forward_rmse,
    backward_rmse  = metrics.backward_rmse,
    symmetric_mae  = metrics.symmetric_mae,
    symmetric_rmse = metrics.symmetric_rmse,
)


@partial(jax.jit, static_argnums=3)
def test_step(batch, state, key, net):
    imagery, mask, snake = prep(batch)
    out = {
        'imagery': imagery,
        'snake': snake,
        'mask': mask,
    }

    # sampled_pred_steps = []
    # subkeys = jax.random.split(key, 4)
    # for subkey in subkeys:
    preds, _ = net(state.params, state.buffers, key, imagery, is_training=False)

    #   sampled_pred_steps.append(pred_steps)
    if isinstance(preds, list):
        # Snake
        out['predictions'] = [preds]
        preds = preds[-1]
    elif preds.shape[:3] == imagery.shape[:3]:
        # Segmentation
        out['segmentation'] = preds
        preds = utils.snakify(preds, snake.shape[-2])
        out['predictions'] = [[preds]]
    else:
        raise ValueError("Model outputs unknown data representation")

    # Convert from normalized to to pixel coordinates
    scale = imagery.shape[1] / 2
    snake *= scale
    preds *= scale

    metrics = {}
    for m in METRICS:
        metrics[m] = jax.vmap(METRICS[m])(preds, snake)

    return metrics, out


if __name__ == '__main__':
    run = Path(sys.argv[1])
    assert run.exists()
    config = yaml.load(open(run / 'config.yml'), Loader=yaml.SafeLoader)

    datasets = ['validation', 'validation_baumhoer', 'validation_zhang']
    loaders  = {d: get_loader(1, 4, d, drop_last=False, subtiles=False) for d in datasets}
    
    for sample_batch in loaders[datasets[0]]:
        break
    img, *_ = prep(sample_batch)

    S, params, buffers = models.get_model(config, img)
    state = utils.load_state(run / 'latest.pkl')
    net = S.apply

    for dataset, loader in loaders.items():
        test_key = jax.random.PRNGKey(0)

        metrics = {k: jnp.zeros([0]) for k in METRICS}
        for batch in tqdm(loader, desc=dataset):
            test_key, subkey = jax.random.split(test_key)
            step_metrics, output = test_step(batch, state, subkey, net)

            metrics = jax.tree_multimap(lambda x, y: jnp.concatenate([x, y]), metrics, step_metrics)

        print('===', dataset, '===')
        print(f'{"Metric".ljust(15)}:    mean   median      min         max')
                                    # 1234567  1234567  1234567  -- 1234567
        with open('eval.csv', 'a') as f:
            for m, val in metrics.items():
                print(f'{m.ljust(15)}: {val.mean():7.4f}  {jnp.median(val):7.4f}  {jnp.min(val):7.4f} -- {jnp.max(val):7.4f}')
                print(f'{run.stem},{config["run_id"]},{dataset},{m},'
                      f'{val.mean()},{jnp.median(val)},{jnp.mean(val/3.33)},{jnp.min(val)},{jnp.max(val)}',
                        file=f)
        print()
