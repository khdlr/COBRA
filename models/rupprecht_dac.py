import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from einops import rearrange 

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


class SimplifiedRupprechtDAC():
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
      direction = jax.vmap(snake_utils.sample_at_vertices, [0, 0])(vertices, flow)
      vertices += self.tau * direction
      steps.append(vertices)

    return {'snake': vertices, 'offsets': flow, 'snake_steps': steps}
