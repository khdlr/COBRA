import jax
import jax.numpy as jnp
import haiku as hk


class UNet:
    def __init__(self, width, depth=4, out_channels=1):
        self.width = width
        self.depth = depth
        self.out_channels = out_channels

    def __call__(self, x, is_training=False):
        skip_connections = []

        W = self.width
        channel_seq = [W * 2**i for i in range(self.depth)]
        for channels in channel_seq:
            x = Convx2(x, channels, is_training)
            skip_connections.append(x)
            x = hk.max_pool(x, 2, 2, padding="SAME")

        x = Convx2(x, 2 * channel_seq[-1], is_training)

        for channels, skip in zip(reversed(channel_seq), reversed(skip_connections)):
            B, H, W, C = x.shape
            B_, H_, W_, C_ = skip.shape

            upsampled = jax.image.resize(x, [B, H_, W_, C], method="bilinear")
            x = hk.Conv2D(C_, 2)(upsampled)
            x = BatchNorm()(x, is_training)
            x = jax.nn.relu(x)
            x = Convx2(jnp.concatenate([x, skip], axis=-1), channels, is_training)

        x = hk.Conv2D(self.out_channels, 1, with_bias=False, w_init=jnp.zeros)(x)
        return {"segmentation": x}


def BatchNorm():
    return hk.BatchNorm(True, True, 0.999)


def Convx2(x, channels, is_training):
    x = hk.Conv2D(channels, 3)(x)
    x = BatchNorm()(x, is_training)
    x = jax.nn.relu(x)
    x = hk.Conv2D(channels, 3)(x)
    x = BatchNorm()(x, is_training)
    x = jax.nn.relu(x)
    return x
