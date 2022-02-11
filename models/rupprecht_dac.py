import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from einops import rearrange 

from .unet import UNet
from . import snake_utils


@jax.vmap
def get_eta(curve):
  grad = jnp.gradient(curve, axis=0)
  eta  = grad @ jnp.array([[0, -1], [1, 0]])
  eta /= jnp.linalg.norm(eta, axis=-1, keepdims=True)
  return eta


class RupprechtDAC():
  def __init__(self, iterations, tau, vertices):
    super().__init__()
    self.iterations = iterations
    self.tau = tau
    self.vertices = vertices

  def __call__(self, imagery, is_training=False):
    x = imagery
    x = hk.max_pool(jax.nn.relu(hk.Conv2D( 32, 3)(x)), 2, 2, 'VALID')
    x = hk.max_pool(jax.nn.relu(hk.Conv2D( 64, 3)(x)), 2, 2, 'VALID')
    x = hk.max_pool(jax.nn.relu(hk.Conv2D(128, 3)(x)), 2, 2, 'VALID')
    x = hk.max_pool(jax.nn.relu(hk.Conv2D(256, 3)(x)), 2, 2, 'VALID')
    x = jax.nn.relu(hk.Conv2D(2048, 1)(x))
    flow = hk.Conv2D(2, 1, w_init=jnp.zeros, with_bias=False)(x)

    init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
    make_bezier = jax.vmap(partial(snake_utils.random_bezier, vertices=self.vertices))
    vertices = make_bezier(init_keys)

    steps = [vertices]
    for _ in range(self.iterations):
      eta       = get_eta(vertices)
      direction = jax.vmap(snake_utils.sample_at_vertices, [0, 0])(vertices, flow)
      alpha     = jnp.einsum('btc,btc->bt', eta, direction)
      alpha = rearrange(alpha, 'b t -> b t 1')
      vertices += self.tau * alpha * eta
      steps.append(vertices)

    return {'snake': vertices, 'offsets': flow, 'snake_steps': steps}


def sobolev_regularization(nu):
  B, L, C = nu.shape
  s = jnp.linspace(0, 1, L, endpoint=False)
  kernel = 1/L * (1 + (jnp.square(s/L) + s/L + 1/6) / (2 * 0.01))
  kernel /= kernel.sum()
  kernel = kernel.reshape(1, 1, L)
  padded = jnp.concatenate([nu, nu[:, :-1]], axis=1)

  nu = rearrange(nu, 'B L C -> (B C) 1 L')
  out = jax.lax.conv(nu, kernel, [1], 'VALID')
  return rearrange(out, '(B C) 1 L -> B L C', B=B, C=C)


class RupprechtUNetDAC():
  def __init__(self, iterations, tau, vertices, sobolev=False):
    super().__init__()
    self.iterations = iterations
    self.tau = tau
    self.vertices = vertices
    self.sobolev = sobolev

  def __call__(self, imagery, is_training=False):
    flow = UNet(32, out_channels=2)(imagery, is_training)['seg']

    init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
    make_bezier = jax.vmap(partial(snake_utils.random_bezier, vertices=self.vertices))
    vertices = make_bezier(init_keys)

    steps = [vertices]
    for _ in range(self.iterations):
      eta       = get_eta(vertices)
      direction = jax.vmap(snake_utils.sample_at_vertices, [0, 0])(vertices, flow)

      if self.sobolev:
        direction = sobolev_regularization(direction)

      alpha     = jnp.einsum('btc,btc->bt', eta, direction)
      alpha = rearrange(alpha, 'b t -> b t 1')
      vertices += self.tau * alpha * eta
      steps.append(vertices)

    return {'snake': vertices, 'offsets': flow, 'snake_steps': steps}

