import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from . import backbones
from . import nnutils as nn
from . import snake_utils


def random_bezier(key, vertices=64):
    t = jnp.linspace(0, 1, vertices).reshape(1, -1, 1)
    points = jax.random.uniform(key, [5, 1, 2], minval=-1, maxval=1)
    while points.shape[0] != 1:
        points = points[1:] * t + points[:-1] * (1-t)
    return points[0]


class RandomDeepSnake():
    def __init__(self, backbone, vertices=64,
            model_dim=64, iterations=5, head='SnakeHead', coord_features=False, stop_grad=True):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.vertices = vertices
        self.stop_grad = stop_grad
        self.head = head

    def __call__(self, imagery, is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        # if is_training:
        feature_maps = [nn.channel_dropout(f, 0.5) for f in feature_maps]

        init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
        make_bezier = jax.vmap(partial(random_bezier, vertices=self.vertices))
        vertices = make_bezier(init_keys)
        steps = [vertices]

        _head = getattr(snake_utils, self.head)(self.model_dim, self.coord_features)
        head = lambda x, y: _head(x, y)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_gradient(vertices)
            vertices = vertices + head(vertices, feature_maps)
            steps.append(vertices)

        return steps
