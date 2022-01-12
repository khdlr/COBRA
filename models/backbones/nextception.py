import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from functools import partial

from .. import nnutils as nn

LayerNorm = partial(hk.LayerNorm, axis=-1, create_scale=True, create_offset=True, eps=1e-6)


class Nextception(hk.Module):
    """Xception backbone like the one used in CALFIN"""
    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape

        # Backbone
        x, skip1 = NextceptionBlock([128, 128, 128], stride=2, return_skip=True)(x)
        x, skip2 = NextceptionBlock([256, 256, 256], stride=2, return_skip=True)(x)
        x, skip3 = NextceptionBlock([768, 768, 768], stride=2, return_skip=True)(x)
        for i in range(8):
            x = NextceptionBlock([768, 768, 768], skip_type='sum', stride=1)(x)

        x = NextceptionBlock([ 728, 1024, 1024], stride=2)(x)
        x = NextceptionBlock([1536, 1536, 2048], stride=1, rate=(1, 2, 4))(x)

        # ASPP
        # Image Feature branch
        bD = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        bD = NextceptionLayer(256)(bD)
        bD = nn.upsample(bD, factor=2)

        b1 = NextceptionLayer(256, rate=1)(x)
        b2 = NextceptionLayer(256, rate=2)(x)
        b3 = NextceptionLayer(256, rate=3)(x)
        b4 = NextceptionLayer(256, rate=4)(x)
        b5 = NextceptionLayer(256, rate=5)(x)
        x  = jnp.concatenate([bD, b1, b2, b3, b4, b5], axis=-1)

        x     = hk.Conv2D(256, 1)(x)
        skip3 = hk.Conv2D(48, 1)(skip3)

        return [skip3, x]


class NextceptionBlock(hk.Module):
    def __init__(self, depth_list, stride, skip_type='conv',
                 rate=1, return_skip=False):
        super().__init__()
        self.blocks = []
        if rate == 1:
            rate = [1, 1, 1]

        for i in range(3):
            self.blocks.append(NextceptionLayer(
                depth_list[i],
                stride=stride if i == 2 else 1,
                rate=rate[i],
            ))

        if skip_type == 'conv':
            self.shortcut = hk.Conv2D(depth_list[-1], 1, stride=stride)
        elif skip_type == 'sum':
            self.shortcut = nn.identity
        self.return_skip = return_skip

    def __call__(self, inputs):
        residual = inputs
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            print(residual.shape)

        shortcut = self.shortcut(inputs)
        outputs = residual + shortcut

        if self.return_skip:
            return outputs, skip
        else:
            return outputs


class NextceptionLayer(hk.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.depthwise = hk.Conv2D(dim, 3, feature_group_count=dim, **kwargs)
        self.norm = LayerNorm()
        self.pointwise = hk.Linear(dim)

    def __call__(self, x):
        x = self.pointwise(x)
        x = jax.nn.gelu(self.norm(x))
        x = self.depthwise(x)
        return x
