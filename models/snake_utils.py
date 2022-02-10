import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from . import nnutils as nn
from functools import partial


def random_bezier(key, vertices=64):
  t = jnp.linspace(0, 1, vertices).reshape(1, -1, 1)
  points = jax.random.uniform(key, [5, 1, 2], minval=-1, maxval=1)
  while points.shape[0] != 1:
    points = points[1:] * t + points[:-1] * (1-t)
  return points[0]


def subdivide_polyline(polyline):
    B, T, C = polyline.shape
    T_new = T * 2 - 1
    resized = jax.vmap(partial(jax.image.resize, shape=(T_new, C), method='linear'))(polyline)
    return resized


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    H, W, C = features.shape
    vertices = (vertices + 1.0) * (jnp.array([[H-1, W-1]]) / 2.0)
    def resample_feature(feature_map: jnp.ndarray):
        return jnd.map_coordinates(feature_map, vertices.T, order=1, mode='constant')

    resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

    return resampled


class SnakeHead(hk.Module):
    def __init__(self, channels, coord_features=False):
        super().__init__()
        self.channels = channels
        self.coord_features = coord_features

    def __call__(self, vertices, feature_maps):
        C = self.channels

        blocks = hk.Sequential([
            hk.Conv1D(C, 1), nn.ReLU(),
            hk.Conv1D(C, 3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=9), nn.ReLU(),
            hk.Conv1D(C, 3, rate=3), nn.ReLU(),
            hk.Conv1D(C, 3), nn.ReLU(),
        ], name='SnakeBlocks')

        # Initialize offset predictors with 0 -> default to no change
        mk_offset = hk.Conv1D(2, 1, with_bias=False, w_init=hk.initializers.Constant(0.0))

        features = []
        for feature_map in feature_maps:
            features.append(jax.vmap(sample_at_vertices, [0, 0])(vertices, feature_map))
        # For coordinate features
        if self.coord_features:
            diff = vertices[:,1:] - vertices[:,:-1]
            diff = jnp.pad(diff, [(0,0), (1,1), (0,0)])
            features.append(diff[:, 1:])
            features.append(diff[:, :-1])
        input_features = jnp.concatenate(features, axis=-1)

        convolved_features = blocks(input_features)
        offsets = mk_offset(convolved_features)
        return offsets


class SeparableSnakeHead(hk.Module):
    def __init__(self, channels, coord_features=False):
        super().__init__()
        self.channels = channels
        self.coord_features = coord_features

    def __call__(self, vertices, feature_maps):
        C = self.channels

        blocks = hk.Sequential([
            nn.SepConvBN1D(C, 3),
            nn.SepConvBN1D(C, 3, rate=3),
            nn.SepConvBN1D(C, 3, rate=9),
            nn.SepConvBN1D(C, 3),
            nn.SepConvBN1D(C, 3, rate=3),
            nn.SepConvBN1D(C, 3, rate=9),
            hk.Conv1D(C, 1),
        ], name='SnakeBlocks')

        # Initialize offset predictors with 0 -> default to no change
        mk_offset = hk.Conv1D(2, 1, with_bias=False, w_init=hk.initializers.Constant(0.0))

        features = []
        for feature_map in feature_maps:
            features.append(jax.vmap(sample_at_vertices, [0, 0])(vertices, feature_map))
        # For coordinate features
        if self.coord_features:
            diff = vertices[:,1:] - vertices[:,:-1]
            diff = jnp.pad(diff, [(0,0), (1,1), (0,0)])
            features.append(diff[:, 1:])
            features.append(diff[:, :-1])
        input_features = jnp.concatenate(features, axis=-1)

        convolved_features = blocks(input_features)
        offsets = mk_offset(convolved_features)
        return offsets


class AuxHead(hk.Module):
    def __init__(self, height, width=None):
        super().__init__()
        self.height = height
        self.width  = width or height

    def __call__(self, feature_maps):
        upscaled = []
        for fm in feature_maps:
            B, H, W, C = fm.shape
            upscaled.append(jax.image.resize(fm, [B, self.height, self.width, C], method='linear'))
        x = jnp.concatenate(upscaled, axis=-1)
        x = jax.nn.relu(hk.Conv2D(32, 1)(x))
        x = hk.Conv2D(2, 1)(x)
        return x

