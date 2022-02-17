import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from . import backbones
from . import nnutils as nn
from . import snake_utils
from .dance import SnakeNet
from .snake_utils import sample_at_vertices, random_bezier


class VanillaDeepSnake():
  def __init__(self, backbone, vertices=64,):
    super().__init__()
    self.backbone = getattr(backbones, backbone)
    self.vertices = vertices

  def __call__(self, imagery, is_training=False):
    backbone = self.backbone()
    features = backbone(imagery, is_training)[-1]

    init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
    make_bezier = jax.vmap(partial(random_bezier, vertices=self.vertices))
    snake = make_bezier(init_keys)
    steps = [snake]

    for _ in range(4):
      # evolve
      deformer = SnakeNet([1, 1, 1, 2, 2, 4, 4])

      sampling_locations = jax.lax.stop_gradient(snake)
      sampled_features = jax.vmap(sample_at_vertices)(sampling_locations, features)
      concat_features = jnp.concatenate([sampled_features, sampling_locations], axis=-1)
      pred_offsets = deformer(concat_features, is_training)
      pred_offsets = jnp.tanh(pred_offsets) * 2

      snake += pred_offsets
      snake = jnp.clip(snake, -1, 1)
      steps.append(snake)

    return {'snake_steps': steps, 'snake': snake}


