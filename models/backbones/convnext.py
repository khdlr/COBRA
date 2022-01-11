import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from functools import partial

from .. import nnutils as nn

LayerNorm = partial(hk.LayerNorm, axis=-1, create_scale=True, create_offset=True, eps=1e-6)

class ConvNeXt(hk.Module):
    """ConvNeXt backbone, adapted from
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"""
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
            return_maps=[1, 3], drop_path_rate=0.1):
        super().__init__()

        self.return_maps = return_maps

        stem = hk.Sequential([
            hk.Conv2D(dims[0], 4, stride=4, padding='VALID'),
            LayerNorm()
        ])
        self.downsample_blocks = [stem]
        for i in range(1, 4):
            down = hk.Sequential([
                LayerNorm(),
                hk.Conv2D(dims[i], 2, stride=2, padding='VALID')
            ])
            self.downsample_blocks.append(down)

        self.stages = []
        dp_rates = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        for depth, dim in zip(depths, dims):
            stage = [ConvNextBlock(dim, dp_rates[cur+i]) for i in range(depth)]
            self.stages.append(stage)
            cur += depth

        self.final_norm = LayerNorm()


    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape
        skips = []
        for down, stage in zip(self.downsample_blocks, self.stages):
            x = down(x)
            for layer in stage:
                x = layer(x, is_training)
            skips.append(x)
        skips[-1] = self.final_norm(x)

        return [skips[m] for m in self.return_maps]


class ConvNextBlock(hk.Module):
    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path

        self.dwconv = hk.Conv2D(dim, 7, feature_group_count=dim)
        self.norm = LayerNorm()
        self.pwconv1 = hk.Linear(4 * dim)
        self.pwconv2 = hk.Linear(dim)

        # self.layer_scale = hk.get_parameter('layer_scale', [dim],
        #         init=lambda shp, dtype: 1e-6 * jnp.ones(shp, dtype))

    def __call__(self, x, is_training):
        assert x.shape[-1] == self.dim

        inputs = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = jax.nn.gelu(x)
        x = self.pwconv2(x)
        # x = self.layer_scale * x

        if is_training:
            x = inputs + nn.sample_dropout(x, self.drop_path)

        return x
