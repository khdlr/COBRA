import jax
import jax.numpy as jnp
import haiku as hk
import optax
import augmax
import numpy as np
from einops import rearrange
from jax.experimental import host_callback
from skimage.measure import find_contours
from functools import partial
from typing import Union, Sequence, Optional, Tuple
from subprocess import check_output
from typing import NamedTuple
import pickle
from inspect import signature


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


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def load_state(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    return state


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


def call_loss(loss_fn, prediction, mask, snake, key='loss', take_mean=True):
    sig = signature(loss_fn)
    args = {'prediction': prediction}
    if 'snake' in sig.parameters:
        args['snake'] = snake
    if 'mask' in sig.parameters:
        args['mask'] = mask

    loss_terms = {}
    if isinstance(prediction, list):
        for i, pred in enumerate(prediction, 1):
            args['prediction'] = pred
            loss_terms[f'{key}_{i}'] = jax.vmap(loss_fn)(**args)
    else:
        loss_terms[key] = jax.vmap(loss_fn)(**args)

    if take_mean:
        loss_terms = jax.tree_map(jnp.mean, loss_terms)

    return loss_terms


def distance_matrix(a, b):
    a = rearrange(a, '(true pred) d -> true pred d', true=1)
    b = rearrange(b, '(true pred) d -> true pred d', pred=1)
    D = jnp.sum(jnp.square(a - b), axis=-1)
    return D


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def fmt_num(x):
    if jnp.isinf(x):
        return '∞'.rjust(8)
    else:
        return f'{x:.2f}'.rjust(8)


def fmt(xs, extra=None):
    tag = ''
    if isinstance(xs, str):
        tag = xs
        xs = extra
    rank = len(xs.shape)
    if rank == 1:
        print(tag, ','.join([fmt_num(x) for x in xs]))
    elif rank == 2:
        print('\n'.join(','.join([fmt_num(x) for x in row]) for row in xs))
        print()


def assert_git_clean():
    diff = check_output(['git', 'diff', 'HEAD'])
    assert not diff, "Won't run on a dirty git state!"


def snakify(mask, vertices):
    res = host_callback.call(snakify_host, (mask, vertices),
            result_shape=jnp.zeros([mask.shape[0], vertices, 2], jnp.float32))
    return res


def snakify_host(args):
    masks, vertices = args
    res = np.zeros([masks.shape[0], vertices, 2], np.float32)
    for i, mask in enumerate(masks):
        contours = find_contours(mask[..., 0], 0)
        # Select the longest contour
        if len(contours) == 0:
            continue

        contour = max(contours, key=lambda x: x.shape[0])
        contour = contour.astype(np.float32)
        contour = contour.view(np.complex64)[:, 0]
        C_space = np.linspace(0, 1, len(contour), dtype=np.float32)
        S_space = np.linspace(0, 1, vertices, dtype=np.float32)
        snake = np.interp(S_space, C_space, contour)
        snake = snake[:, np.newaxis].view(np.float64).astype(np.float32)

        res[i] = snake * (2.0 / mask.shape[0]) - 1.0
    return res


def _infer_shape(
    x: jnp.ndarray,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> Tuple[int, ...]:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return (1,) + tuple(size if d != channel_axis else 1
                        for d in range(1, x.ndim))
  elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
    return (1,) * (x.ndim - len(size)) + tuple(size)
  else:
    assert x.ndim == len(size)
    return tuple(size)


def min_pool(
    value: jnp.ndarray,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str = "SAME",
    channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
  """Min pool.
  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, an int or same rank as value.
    strides: Strides of the pooling window, an int or same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped,
      used to infer ``window_shape`` or ``strides`` if they are an integer.
  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return jax.lax.reduce_window(value, jnp.inf, jax.lax.min, window_shape, strides, padding)

