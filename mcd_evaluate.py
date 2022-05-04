import sys
import yaml
import json
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import tqdm
from PIL import Image

import models
import haiku as hk
from data_loading import get_loader
from lib import losses, utils, logging
from lib.utils import TrainingState, prep, load_state
from models.nnutils import channel_dropout


MONKEY_PATCHED = True
# Monkey-Patch Haiku Convs to get MC dropout
class MCDConv2D(hk.ConvND):
  def __init__(
      self,
      output_channels,
      kernel_shape,
      stride=1,
      rate=1,
      padding='SAME',
      with_bias: bool = True,
      w_init=None,
      b_init=None,
      data_format: str = "NHWC",
      mask=None,
      feature_group_count: int = 1,
      name=None,
  ):
    super().__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        feature_group_count=feature_group_count,
        name=name)

  def __call__(self, inputs: jnp.ndarray, *,
      precision=None,):
    x = super().__call__(inputs, precision=precision)
    if MONKEY_PATCHED:
      print("Dropping Out")
      x = channel_dropout(x, rate=0.5)
    return x

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
      terms['snake'] = utils.snakify(terms['segmentation'], contour.shape[-2])
    if 'snake_steps' not in terms:
      terms['snake_steps'] = [terms['snake']]

    # Convert from normalized to to pixel coordinates
    scale = imagery.shape[1] / 2
    for key in ['snake', 'snake_steps', 'contour']:
      terms[key] = jax.tree_map(lambda x: scale * (1.0 + x), terms[key])

    metrics = {}
    for m in METRICS:
        metrics[m] = losses.call_loss(METRICS[m], terms)[0]

    return metrics, terms


if __name__ == '__main__':
    run = Path(sys.argv[1])
    assert run.exists()
    do_output = True

    config = yaml.load(open(run / 'config.yml'), Loader=yaml.SafeLoader)
    if 'dataset' in config and config['dataset'] == 'TUD-MS':
      # datasets = ['TEST' , '', 'validation_zhang']
      loaders  = {'TUD-MS': get_loader(4, 1, 'test', config, None, subtiles=False)}
    else:
      config['dataset'] = 'CALFIN'
      config['data_root'] = '../CALFIN/training/data'
      config['data_channels'] = [2]

      datasets = ['validation' , 'validation_baumhoer', 'validation_zhang']
      loaders  = {d: get_loader(4, 1, d, config, None, subtiles=False) for d in datasets}

      config['dataset'] = 'TUD'
      config['data_root'] = '../aicore/uc1/data/'
      config['data_channels'] = ['SPECTRAL/BANDS/STD_2s_B8_8b']
      loaders['TUD_test'] = get_loader(4, 1, 'test', config, subtiles=False)

    for sample_batch in list(loaders.values())[0]:
      img, *_ = prep(sample_batch)
      break

    hk.Conv2D = Conv2D
    S, params, buffers = models.get_model(config, img)
    state = utils.load_state(run / 'latest.pkl')
    net = S.apply

    S_mc, params, buffers = models.get_model(config, img)
    net_mc = S_mc.apply

    img_root = run / 'imgs_mcd'
    img_root.mkdir(exist_ok=True)

    all_metrics = {}

    output_acc = []
    samples_acc = []
    for dataset, loader in loaders.items():
        test_key = jax.random.PRNGKey(27)
        test_metrics = {}

        img_dir = img_root / dataset
        img_dir.mkdir(exist_ok=True)
        dsidx = 0
        for batch in tqdm(loader, desc=dataset):
            test_key, *subkeys = jax.random.split(test_key, 11)

            MONKEY_PATCHED = False
            metrics, output = test_step(batch, state, test_key, net)
            all_samples = []
            MONKEY_PATCHED = True
            for k in subkeys:
              _, out = test_step(batch, state, k, net_mc)
              all_samples.append(out['snake'])

            for m in metrics:
              if m not in test_metrics: test_metrics[m] = []
              test_metrics[m].append(metrics[m])

            # for i in range(len(output['imagery'])):
            #   samples = jax.tree_map(lambda x: x[i], all_samples)
            #   o = jax.tree_map(lambda x: x[i], output)

            #   raw = Image.fromarray((255 * np.asarray(o['imagery'][..., 0])).astype(np.uint8))
            #   base = 0.5 * (o['imagery'] + 1.0)

            #   logging.draw_multi(base, o['contour'],
            #       samples, img_dir / f'{dsidx:03d}_samples.pdf')
            #   logging.draw_uncertainty(base, o['contour'],
            #       o['snake'], samples, img_dir / f'{dsidx:03d}_std.pdf')

            #   dsidx += 1

            output_acc.append(output)
            samples_acc.append(jnp.stack(all_samples, axis=1))

        logging.log_metrics(test_metrics, dataset, 0, do_wandb=False)
        for m in test_metrics:
            all_metrics[f'{dataset}/{m}'] = np.mean(test_metrics[m])

        full_output  = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *output_acc)
        for drop in ['segmentation', 'mask', 'edge', 'snake_steps']:
          if drop in full_output:
            del full_output[drop]
        full_samples = jnp.concatenate(samples_acc, axis=0)
        np.savez(run / f'{dataset}.npz', **full_output, samples=full_samples)

    with (run / 'uncertainty_metrics.json').open('w') as f:
        print(all_metrics, file=f)
