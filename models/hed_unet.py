import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange


class HEDUNet:
    def __init__(self, width=16, depth=6):
        self.width = width
        self.depth = depth

    def __call__(self, x, is_training=False):
        B, H, W, C = x.shape
        # init
        inputs = x
        x = hk.Conv2D(self.width, 1)(x)

        # Contracting Path
        skip_connections = []
        for i in range(self.depth):
            skip_connections.append(x)
            x = DownBlock(x, is_training)

        # Expanding Path
        multilevel_features = [x]
        for skip in reversed(skip_connections):
            x = UpBlock(x, skip, is_training)
            multilevel_features.append(x)

        Ps = []
        Qs = []
        for feature_map in reversed(multilevel_features):
            p = hk.Conv2D(2, 1)(feature_map)
            q = hk.Conv2D(2, 1)(feature_map)
            Ps.append(jax.image.resize(p, [B, H, W, 2], "bilinear"))
            Qs.append(jax.image.resize(q, [B, H, W, 2], "bilinear"))

        P = jnp.concatenate(Ps, axis=-1)
        Q = jnp.concatenate(Qs, axis=-1)

        A = rearrange(Q, "B H W (R C) -> B H W C R", C=2)
        P_ = rearrange(P, "B H W (R C) -> B H W C R", C=2)
        final_pred = jnp.einsum("bhwcr,bhwcr->bhwc", jax.nn.softmax(A, axis=-1), P_)
        all_preds = jnp.concatenate([final_pred, P], axis=-1)

        seg, edge = jnp.split(final_pred, 2, axis=-1)

        return {"segmentation": seg, "edge": edge, "hed_unet_stack": all_preds}


def BatchNorm():
    return hk.BatchNorm(True, True, 0.999)


def Convx2(x, channels, is_training):
    x = hk.Conv2D(channels, 3, with_bias=False)(x)
    x = BatchNorm()(x, is_training)
    x = jax.nn.relu(x)
    x = hk.Conv2D(channels, 3, with_bias=False)(x)
    x = BatchNorm()(x, is_training)
    x = jax.nn.relu(x)
    return x


def DownBlock(x, is_training):
    C = x.shape[-1] * 2
    x = hk.Conv2D(C, 2, stride=2, with_bias=False)(x)
    x = jax.nn.relu(BatchNorm()(x, is_training))
    x = Convx2(x, C, is_training)
    return x


def UpBlock(x, skip, is_training):
    C = x.shape[-1] // 2
    x = hk.Conv2DTranspose(C, 2, stride=2, with_bias=False)(x)
    x = jax.nn.relu(BatchNorm()(x, is_training))
    x = jnp.concatenate([x, skip], axis=-1)
    x = Convx2(x, C, is_training)
    return x
