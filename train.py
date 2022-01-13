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
from lib import utils, losses, metrics, logging
from lib.utils import TrainingState, prep, changed_state, save_state
from evaluate import test_step, METRICS


PATIENCE = 100

def get_optimizer():
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=1e-3,
        warmup_steps=10 * 487,
        decay_steps=(500-10) * 487,
        end_value=1e-5
    )
    return optax.adam(lr_schedule, b1=0.9, b2=0.99)


@partial(jax.jit, static_argnums=3)
def train_step(batch, state, key, net):
    _, optimizer = get_optimizer()

    aug_key, model_key = jax.random.split(key)
    img, mask, snake = prep(batch, aug_key, augment=True)

    def calculate_loss(params):
        preds, buffers = net(params, state.buffers, model_key, img, is_training=True)
        loss_terms = utils.call_loss(loss_fn, preds, mask, snake)

        if isinstance(preds, list):
            preds = preds[-1]

        return sum(loss_terms.values()), (buffers, preds, loss_terms)

    (loss, (buffers, prediction, metrics)), gradients = jax.value_and_grad(calculate_loss, has_aux=True)(state.params)
    updates, new_opt = optimizer(gradients, state.opt, state.params)
    new_params = optax.apply_updates(state.params, updates)

    if prediction.ndim > 3:
        prediction = utils.snakify(prediction[:1], snake.shape[-2])
        snake = snake[:1]

    # Convert from normalized to to pixel coordinates
    scale = img.shape[1] / 2
    snake *= scale
    prediction *= scale

    for m in METRICS:
        metrics.update(utils.call_loss(METRICS[m], prediction, mask, snake, key=m))

    return metrics, changed_state(state,
        params=new_params,
        buffers=buffers,
        opt=new_opt,
    )


if __name__ == '__main__':
    utils.assert_git_clean()
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)

    config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
    lf = config['loss_function']
    if lf.endswith(')'):
        lf_name, lf_args = lf[:-1].split('(')
        loss_cls = getattr(losses, lf_name)
        if lf_args: 
            lf_args = yaml.load(f'[{lf_args}]', Loader=yaml.SafeLoader)
            loss_fn = loss_cls(*lf_args)
        else:
            loss_fn = loss_cls()
    else:
        loss_fn = getattr(losses, lf)

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    train_loader = get_loader(config['batch_size'], 4, 'train', subkey)
    val_loader   = get_loader(4, 1, 'validation', None, subtiles=False)

    img, *_ = prep(next(iter(train_loader)))
    S, params, buffers = models.get_model(config, img)

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))
    net = S.apply

    running_min = np.inf
    last_improvement = 0
    wandb.init(project='Deep Snake', config=config)

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
            metrics, state = train_step(batch, state, subkey, net)

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
            imagery     = out['imagery'][0]
            snake       = out['snake'][0]
            predictions = [p[0] for p in out['predictions']]
            logging.log_anim(out, f"Animated/{step}", epoch)
            if 'segmentation' in out:
                segmentation = out['segmentation'][0]
                mask = out['mask'][0]
                logging.log_segmentation(imagery, mask, segmentation, f'Segmentation/{step}', epoch)

        logging.log_metrics(val_metrics, 'val', epoch)
