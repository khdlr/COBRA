"""
This code is a JAX port of
https://github.com/lkevinzc/dance/blob/master/core/modeling/edge_snake/dance.py
"""
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from . import backbones
from . import nnutils as nn
from .snake_utils import sample_at_vertices, random_bezier

from functools import partial
from einops import rearrange, repeat

init = dict(
  w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'),
)

class DANCE():
  def __init__(self, backbone='ResNet50', vertices=64):
    super().__init__()
    self.backbone = getattr(backbones, backbone)
    self.refinement_head = EdgeSnakeFPNHead()
    self.snake_head = SnakeHead(vertices)

  def __call__(self, imagery, is_training=False):
    backbone = self.backbone()
    features = backbone(imagery, is_training)
    snake_input, edge_map = self.refinement_head(features, is_training)
    snake_res = self.snake_head(snake_input, edge_map, is_training)
    return {**snake_res, 'edge': edge_map}


class EdgeSnakeFPNHead():
  def __call__(self, features, is_training):
    max_H = max(f.shape[1] for f in features)

    # Merge Features
    merged_features = 0.
    for feat in features:
      x = feat
      while x.shape[1] < max_H:
        x = hk.Conv2D(128, 3, with_bias=False, **init)(x)
        x = jax.nn.relu(hk.GroupNorm(32)(x))
        B, H, W, C = x.shape
        x = jax.image.resize(x, [B, 2*H, 2*W, C], 'bilinear')
      x = hk.Conv2D(128, 3, with_bias=False, **init)(x)
      x = jax.nn.relu(hk.GroupNorm(32)(x))
      merged_features += x

    # Predictor
    x = hk.Conv2D(128, 3, with_bias=False, **init)(merged_features)
    x = jax.nn.relu(hk.GroupNorm(32)(x))
    pred_logits = hk.Conv2D(1, 1, **init)(x)
    pred_edge = jax.nn.sigmoid(pred_logits)

    # Attender
    a = 1 - pred_edge
    # conv1
    a = hk.Conv2D(32, 3, with_bias=False, **init)(a)
    a = jax.nn.relu(hk.GroupNorm(4)(a))
    # conv2
    a = hk.Conv2D(1, 3, w_init=hk.initializers.TruncatedNormal(0.01))(a)
    a = jax.nn.sigmoid(a)
    # conv3

    return merged_features, a


class SnakeBlock(hk.Module):
  def __init__(self, out_dim, n_adj=4, rate=1):
    super().__init__()
    self.out_dim = out_dim
    self.n_adj = n_adj
    self.rate = rate

  def __call__(self, x, is_training):
    x = hk.Conv1D(self.out_dim, 2*self.n_adj+1, rate=self.rate)(x)
    x = jax.nn.relu(x)
    x = hk.BatchNorm(True, True, 0.9)(x, is_training)
    return x


class SnakeNet(hk.Module):
  def __init__(self, dilations):
    super().__init__()
    self.dilations = dilations

  def __call__(self, x, is_training):
    x = SnakeBlock(128)(x, is_training)  # self.head

    states = [x]
    for rate in self.dilations:
      x = SnakeBlock(128, rate=rate)(x, is_training)
      states.append(x)

    state = jnp.concatenate(states, axis=-1)

    back_out = hk.Conv2D(256, 1)(x)  # self.fusion
    global_state = jnp.max(back_out, axis=1, keepdims=True)
    global_state = repeat(global_state, 'B 1 C -> B T C', T=state.shape[1])
    state = jnp.concatenate([global_state, state], axis=-1)

    # self.prediction
    x = state
    x = hk.Conv1D(256, 1)(x)
    x = jax.nn.relu(x)
    x = hk.Conv1D(64, 1)(x)
    x = jax.nn.relu(x)
    x = hk.Conv1D(2, 1)(x)

    return x



class SnakeHead():
  def __init__(self, vertices):
    self.vertices = vertices

  def __call__(self, features, edge_band, is_training):
    #TODO
    # bottom_out
    features = hk.Conv2D(128, 3, with_bias=False)(features)
    features = jax.nn.relu(hk.GroupNorm(32)(features))
    features = hk.Conv2D(128, 3, with_bias=False)(features)
    features = jax.nn.relu(hk.GroupNorm(32)(features))

    # Snake deformation
    features = jnp.concatenate([edge_band, features], axis=-1)

    init_keys = jax.random.split(hk.next_rng_key(), features.shape[0])
    make_bezier = jax.vmap(partial(random_bezier, vertices=self.vertices))
    snake = make_bezier(init_keys)

    steps = [snake]
    for _ in range(3):
      # evolve
      deformer = SnakeNet([1, 1, 1, 2, 2, 4, 4])

      sampling_locations = jax.lax.stop_gradient(snake)
      sampled_features = jax.vmap(sample_at_vertices)(sampling_locations, features)

      att_scores = sampled_features[..., :1]
      sampled_features = sampled_features[..., 1:]
      loc_features = de_location(sampling_locations)
      concat_features = jnp.concatenate([sampled_features, loc_features], axis=-1)
      pred_offsets = deformer(concat_features, is_training)
      pred_offsets = jnp.tanh(pred_offsets) * 2
      pred_offsets = pred_offsets * att_scores

      snake += pred_offsets
      snake = jnp.clip(snake, -1, 1)
      steps.append(snake)

    return {'snake_steps': steps, 'snake': snake}



def de_location(locations, eps=0.001):
  mins = jnp.min(locations, axis=-1, keepdims=True)
  maxs = jnp.max(locations, axis=-1, keepdims=True)

  return (locations - mins) / jnp.maximum(maxs - mins, eps) 
