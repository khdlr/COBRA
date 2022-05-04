import yaml
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from data_loading import get_loader 
from functools import partial

import wandb
from tqdm import tqdm

import sys
import augmax

import models
from lib import utils, losses, logging
from lib.utils import TrainingState, prep, changed_state, save_state
from mcd_evaluate import test_step, METRICS, MCDConv2D
hk.Conv2D = MCDConv2D

from jax.config import config
config.update("jax_numpy_rank_promotion", "raise")


def get_optimizer():
  lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-7,
    peak_value=1e-3,
    warmup_steps=10 * 487,
    decay_steps=(500-10) * 487,
    end_value=4e-5
  )
  return optax.adam(lr_schedule, b1=0.5, b2=0.99)


@partial(jax.jit, static_argnums=3)
def train_step(batch, state, key, net):
    _, optimizer = get_optimizer()

    aug_key, model_key = jax.random.split(key)
    img, mask, contour = prep(batch, aug_key, augment=True)

    def calculate_loss(params):
        terms, buffers = net(params, state.buffers, model_key, img, is_training=True)
        terms = {**terms, 'mask': mask, 'contour': contour}
        loss, loss_terms = losses.call_loss(loss_fn, terms)

        return loss, (buffers, terms, loss_terms)

    (loss, (buffers, terms, metrics)), gradients = jax.value_and_grad(calculate_loss, has_aux=True)(state.params)
    updates, new_opt = optimizer(gradients, state.opt, state.params)
    new_params = optax.apply_updates(state.params, updates)

    terms = {
        **terms,
        'mask': mask,
        'contour': contour,
        'imagery': img,
    }

    if 'snake' not in terms:
      terms['snake'] = utils.snakify(terms['segmentation'][:1], contour.shape[-2])
      terms['contour'] = terms['contour'][:1]
    if 'snake_steps' not in terms:
      terms['snake_steps'] = [terms['snake']]

    # Convert from normalized to to pixel coordinates
    scale = img.shape[1] / 2
    for key in ['snake', 'snake_steps', 'contour']:
      terms[key] = jax.tree_map(lambda x: scale * (1.0 + x), terms[key])

    for m in METRICS:
        metrics[m] = jnp.mean(losses.call_loss(METRICS[m], terms)[0])

    return metrics, terms, changed_state(state,
        params=new_params,
        buffers=buffers,
        opt=new_opt,
    )


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != '-f':
        utils.assert_git_clean()
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)

    config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
    # Don't do this at home, kids!
    loss_fn = eval(config['loss_function'], losses.__dict__)

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    B = config['batch_size']
    train_loader = get_loader(B, 4, 'train', config, subkey)
    val_loader   = get_loader(4, 1, 'validation', config, None, subtiles=False)

    img, *_ = prep(next(iter(train_loader)))
    S, params, buffers = models.get_model(config, img)

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))
    net = S.apply

    running_min = np.inf
    last_improvement = 0
    wandb.init(project=f'Deep Snake MCD {config["dataset"]}', config=config)

    run_dir = Path(f'runs/{wandb.run.id}/')
    run_dir.mkdir(parents=True)
    config['run_id'] = wandb.run.id
    with open(run_dir / 'config.yml', 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

    for epoch in range(1, 501):
        wandb.log({f'epoch': epoch}, step=epoch)
        prog = tqdm(train_loader, desc=f'Ep {epoch} Trn')
        trn_metrics = {}
        loss_ary = None
        for step, batch in enumerate(prog, 1):
            train_key, subkey = jax.random.split(train_key)
            metrics, terms, state = train_step(batch, state, subkey, net)

            for m in metrics:
              if m not in trn_metrics: trn_metrics[m] = []
              trn_metrics[m].append(metrics[m])

        logging.log_metrics(trn_metrics, 'trn', epoch, do_print=False)

        if epoch % 10 != 0:
             continue

        # Save Checkpoint
        save_state(state, run_dir / f'latest.pkl')

        # Validate
        val_key = persistent_val_key
        val_metrics = {}
        for step, batch in enumerate(val_loader):
            val_key, subkey = jax.random.split(val_key)
            metrics, out = test_step(batch, state, subkey, net)

            for m in metrics:
              if m not in val_metrics: val_metrics[m] = []
              val_metrics[m].append(metrics[m])

            out = jax.tree_map(lambda x: x[0], out) # Select first example from batch
            logging.log_anim(out, f"Animated/{step}", epoch)
            if 'segmentation' in out:
                logging.log_segmentation(out, f'Segmentation/{step}', epoch)
            if 'offsets' in out:
                logging.log_offset_field(out, f'Offsets/{step}', epoch)
            if 'edge' in out:
                logging.log_edge(out, f'Edge/{step}', epoch)

        logging.log_metrics(val_metrics, 'val', epoch)
