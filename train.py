import yaml
from typing import NamedTuple, Any
import pickle
from pathlib import Path
from inspect import signature

import numpy as np
import jax
import jax.numpy as jnp
from jax.profiler import TraceAnnotation
from jax.experimental.host_callback import id_print
import haiku as hk
import optax
from data_loading import get_loader 
from generate_data import generate_image 
from functools import partial

import wandb
from tqdm import tqdm

import sys
import augmax

import models
import utils
import loss_functions
from plotting import log_segmentation, log_anim


METRICS = dict(
    l1 = loss_functions.l1_loss,
    l2 = loss_functions.l2_loss,
    min_min = loss_functions.min_min_loss,
    dtw = loss_functions.DTW(),
    forward_mae = loss_functions.forward_mae,
    backward_mae = loss_functions.backward_mae,
    forward_rmse = loss_functions.forward_rmse,
    backward_rmse = loss_functions.backward_rmse,
)
PATIENCE = 100


def get_optimizer():
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=1e-3,
        warmup_steps=1024,
        decay_steps=200000-1024,
        end_value=1e-5
    )
    return optax.adam(lr_schedule, b1=0.5, b2=0.9)


class TrainingState(NamedTuple):
    params: hk.Params
    buffers: hk.State
    opt: optax.OptState


def changed_state(state, params=None, buffers=None, opt=None):
    return TrainingState(
        params = state.params if params is None else params,
        buffers = state.buffers if buffers is None else buffers,
        opt = state.opt if opt is None else opt,
    )


def prep(batch, key=None, augment=False, input_types=None):
    ops = []
    if augment: ops += [
        augmax.HorizontalFlip(),
        augmax.VerticalFlip(),
        augmax.Rotate90(),
        augmax.Rotate(15),
        # augmax.Warp(coarseness=16)
    ]
    ops += [augmax.ByteToFloat()]
    # if augment: ops += [
    #     augmax.ChannelShuffle(p=0.1),
    #     augmax.Solarization(p=0.1),
    # ]

    if input_types is None:
        input_types = [
            augmax.InputType.IMAGE,
            augmax.InputType.MASK,
            augmax.InputType.CONTOUR,
        ]
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment == False:
        key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, batch[0].shape[0])
    transformation = jax.vmap(chain)
    outputs = list(transformation(subkeys, *batch))
    for i, typ in enumerate(input_types):
        if typ == augmax.InputType.CONTOUR:
            outputs[i] = 2 * (outputs[i] / outputs[0].shape[1]) - 1.0

    return outputs


@partial(jax.jit, static_argnums=3)
def train_step(batch, state, key, net):
    _, optimizer = get_optimizer()

    aug_key, model_key = jax.random.split(key)
    img, mask, snake = prep(batch, aug_key, augment=True)

    def calculate_loss(params):
        preds, buffers = net(params, state.buffers, model_key, img, is_training=True)
        loss_terms = loss_functions.call_loss(loss_fn, preds, mask, snake)

        if isinstance(preds, list):
            preds = preds[-1]

        return sum(loss_terms.values()), (buffers, preds, loss_terms)

    (loss, (buffers, prediction, metrics)), gradients = jax.value_and_grad(calculate_loss, has_aux=True)(state.params)
    updates, new_opt = optimizer(gradients, state.opt, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    if prediction.ndim > 3:
        prediction = utils.snakify(prediction[:1], snake.shape[-2])
        snake = snake[:1]

    for m in METRICS:
        metrics.update(loss_functions.call_loss(METRICS[m], prediction, mask, snake, key=m))

    return metrics, changed_state(state,
        params=new_params,
        buffers=buffers,
        opt=new_opt,
    )


@partial(jax.jit, static_argnums=3)
def val_step(batch, state, key, net):
    imagery, mask, snake = prep(batch)

    sampled_pred_steps = []
    subkeys = jax.random.split(key, 4)
    for subkey in subkeys:
        pred_steps, _ = net(state.params, state.buffers, subkey, imagery, is_training=False)
        sampled_pred_steps.append(pred_steps)

    metrics = loss_functions.call_loss(loss_fn, sampled_pred_steps[0], mask, snake)
    preds = sampled_pred_steps[0][-1]

    out = {
        'imagery': imagery,
        'snake': snake,
        'mask': mask,
    }

    # TODO: Random sampling probably broke this following code for
    # segmentation based models
    if preds.ndim > 3:
        out['segmentation'] = preds
        preds = utils.snakify(preds, snake.shape[-2])
        sampled_pred_steps = [preds]
    out['predictions'] = sampled_pred_steps

    for m in METRICS:
        metrics.update(loss_functions.call_loss(METRICS[m], preds, mask, snake, key=m))

    return metrics, out

def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def log_metrics(metrics, prefix, epoch, do_print=True):
    metrics = {m: np.mean(metrics[m]) for m in metrics}

    wandb.log({f'{prefix}/{m}': metrics[m] for m in metrics}, step=epoch)
    if do_print:
        print(f'{prefix}/metrics')
        print(', '.join(f'{k}: {v:.3f}' for k, v in metrics.items()))


if __name__ == '__main__':
    # utils.assert_git_clean()
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)
    multiplier = 64

    config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
    lf = config['loss_function']
    if lf.endswith(')'):
        lf_name, lf_args = lf[:-1].split('(')
        loss_cls = getattr(loss_functions, lf_name)
        if lf_args: 
            lf_args = yaml.load(f'[{lf_args}]', Loader=yaml.SafeLoader)
            loss_fn = loss_cls(*lf_args)
        else:
            loss_fn = loss_cls()
    else:
        loss_fn = getattr(loss_functions, lf)

    model_args = config['model_args']
    modelclass = getattr(models, config['model'])
    if 'vertices' in signature(modelclass).parameters:
        model_args['vertices'] = config['vertices']
    S = modelclass(**model_args)
    S = hk.transform_with_state(S)

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    train_loader = get_loader(config['batch_size'], 4, 'train', subkey)
    val_loader   = get_loader(16, 1, 'validation', None)
    img, *_ = prep(next(iter(train_loader)))

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()

    params, buffers = S.init(jax.random.PRNGKey(39), img[:1], is_training=True)
    state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))
    net = S.apply

    running_min = np.inf
    last_improvement = 0
    wandb.init(project='DeepSnake CALFIN', config=config)

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

        log_metrics(trn_metrics, 'trn', epoch, do_print=False)

        if epoch % 10 != 0:
            continue

        # Save Checkpoint
        ckpt_dir = Path('checkpoints')
        ckpt_dir.mkdir(exist_ok=True)
        save_state(state, ckpt_dir / f'{wandb.run.id}-latest.npz')

        # Validate
        val_key = persistent_val_key
        val_metrics = {}
        for step, batch in enumerate(val_loader):
            val_key, subkey = jax.random.split(val_key)
            metrics, out = val_step(batch, state, subkey, net)

            for m in metrics:
              if m not in val_metrics: val_metrics[m] = []
              val_metrics[m].append(metrics[m])

            out = jax.tree_map(lambda x: x[0], out) # Select first example from batch
            imagery     = out['imagery'][0]
            snake       = out['snake'][0]
            predictions = [p[0] for p in out['predictions']]
            log_anim(out, f"Animated/{step}", epoch)
            if 'segmentation' in out:
                segmentation = out['segmentation'][0]
                mask = out['mask'][0]
                log_segmentation(imagery, mask, segmentation, f'Segmentation/{step}', epoch)

        log_metrics(val_metrics, 'val', epoch)
