import jax
import jax.numpy as jnp
import haiku as hk

from .backbones import CALFINXception
from . import nnutils as nn


class CFM:
    """A port of the CALFIN-DeepLabv3 with xception backbone,
    more specifically, model_cfm_dual_wide_x65.py
    """
    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape
        skip3, x = CALFINXception()(x, is_training)

        # Decoder
        x = nn.upsample(x, shp=[H//4, W//4])
        dec_skip1 = nn.ConvBNAct(48, 1, act='elu')(skip3, is_training)
        x = jnp.concatenate([x, dec_skip1], axis=-1)
        x = nn.SepConvBN(256, depth_activation=True)(x, is_training)
        x = nn.SepConvBN(256, depth_activation=True)(x, is_training)
        x = hk.Conv2D(2, 1)(x)
        x = nn.upsample(x, shp=[H, W])

        return x
