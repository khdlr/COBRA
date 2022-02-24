"""
This code is a port of
https://github.com/dmarcosg/DSAC/blob/master/snake_utils.py
to JAX
"""
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from einops import rearrange 

from .unet import UNet
from . import snake_utils

from jax.experimental.host_callback import id_print
def checknan(tensor, tag):
  id_print(jnp.isnan(tensor).any(), tag=tag)

def active_contour_step(F, alpha, B, snake, d, gamma, max_px_move):
  L, C = snake.shape

  sample_snake = jax.lax.stop_gradient(snake)
  f = snake_utils.sample_at_vertices(sample_snake, F)
  beta = snake_utils.sample_at_vertices(sample_snake, B)[..., 0]

  # We need to adapt the ACM iteration for an open polyline
  # instead of a closed polygon
  def E_int(snake):
    snake_  = jnp.gradient(snake, axis=0)
    snake__ = jnp.gradient(snake_, axis=0)

    membrane_term   = jnp.sum(alpha * jnp.sum(jnp.square(snake_), axis=-1))
    thin_plate_term = jnp.sum(beta * jnp.sum(jnp.square(snake__), axis=-1))
    return membrane_term + thin_plate_term

  # def E_b(snake):
  #   t = jnp.linspace(0, 1, 10, endpoint=False).reshape(-1, 1, 1)
  #   snake_l = snake[ :-1].reshape(1, L-1, C)
  #   snake_r = snake[1:  ].reshape(1, L-1, C)

  #   # snake_r|l: 1xLxC
  #   # t: Tx1x1
  #   sample_points = (1-t) * snake_r + t * snake_r
  #   sample_points = sample_points.reshape(-1, C)
  #   kappas = snake_utils.sample_at_vertices(sample_points, kappa)
  #   return jnp.mean(kappas)

  # d = -0.5 * max_px_move * jnp.tanh(f - jax.grad(E_b)(snake)*gamma) + 0.5 * d
  d = -0.5 * max_px_move * jnp.tanh(f) + 0.5 * d
  snake = snake + gamma * d - jax.grad(E_int)(snake) * gamma

  snake = jnp.clip(snake, -1, 1)

  # FIXME: PLACEHOLDER
  return (snake, d), snake


class MarcosDSAC():
  def __init__(self, iterations=64, vertices=64):
    super().__init__()
    self.iterations = iterations
    self.vertices = vertices

  def __call__(self, imagery, is_training=False):
    features = UNet(32, out_channels=3)(imagery, is_training)['segmentation']

    D, alpha, beta = jnp.split(features, 3, axis=-1)
    F_y = jnp.gradient(D, axis=1)
    F_x = jnp.gradient(D, axis=2)
    F = jnp.concatenate([F_y, F_x], axis=-1)

    alpha = jnp.mean(alpha[..., 0], axis=[1, 2])

    init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
    make_bezier = jax.vmap(partial(snake_utils.random_bezier, vertices=self.vertices))
    snake = make_bezier(init_keys)

    # build snake step function
    step_fn_raw = partial(active_contour_step, gamma=1.0, max_px_move=0.2)
    step_fn_vmapped = jax.vmap(step_fn_raw)
    step_fn = lambda snake_d, i: step_fn_vmapped(F, alpha, beta, snake_d[0], snake_d[1])

    init = (snake, jnp.zeros_like(snake))

    _, steps = jax.lax.scan(step_fn, init, None, length=self.iterations)
    steps = list(steps)

    return {'snake': steps[-1], 'snake_steps': steps, 'mapE': D, 'mapA': alpha, 'mapB': beta}

