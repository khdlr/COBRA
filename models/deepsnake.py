import jax
import jax.numpy as jnp
import haiku as hk

from . import backbones
from . import nnutils as nn
from . import snake_utils


class DeepSnake():
    def __init__(self, backbone, vertices=64,
            model_dim=64, iterations=5, head='SnakeHead', coord_features=False,
            weight_sharing=True, stop_grad=True):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.vertices = vertices
        self.stop_grad = stop_grad
        self.weight_sharing = weight_sharing
        self.head = head

    def __call__(self, imagery, is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        if is_training:
            feature_maps = [nn.channel_dropout(f, 0.5) for f in feature_maps]

        vertices = jnp.zeros([imagery.shape[0], self.vertices, 2])
        steps = [vertices]

        if self.weight_sharing:
          _head = getattr(snake_utils, self.head)(self.model_dim, self.coord_features)
          head = lambda x, y: _head(x, y)
        else:
          head_cls = getattr(snake_utils, self.head)
          head = lambda x, y: head_cls(self.model_dim, self.coord_features)(x, y)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_gradient(vertices)
            vertices = vertices + head(vertices, feature_maps)
            steps.append(vertices)

        return steps


class PerturbedDeepSnake(DeepSnake):
    def __call__(self, imagery, is_training=False):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training)

        # if is_training:
        feature_maps = [nn.channel_dropout(f, 0.5) for f in feature_maps]

        mkrand = lambda: 0.01 * jax.random.normal(hk.next_rng_key(), [imagery.shape[0], self.vertices, 2])
        vertices = mkrand()
        steps = [vertices]

        _head = SnakeHead(self.model_dim, self.coord_features)
        head = lambda x, y: _head(x, y)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_gradient(vertices)
            vertices = vertices + head(vertices, feature_maps) + mkrand()
            steps.append(vertices)

        return steps
