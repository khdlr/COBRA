import jax
import jax.numpy as jnp
import haiku as hk

from .. import hed_unet as unet


class ResBlock(hk.Module):
    def __init__(self, dim, inner_dim=32):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        init = hk.Conv2D(self.inner_dim, 1)
        conv1 = hk.Conv2D(self.inner_dim, 3)
        conv2 = hk.Conv2D(self.inner_dim, 3)
        post = hk.Conv2D(self.dim, 1)

        skip = x
        x = init(jax.nn.relu(x))
        x = conv1(jax.nn.relu(x))
        x = conv2(jax.nn.relu(x))
        x = post(jax.nn.relu(x))
        return skip + x


class SimpleBackbone(hk.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def __call__(self, x, is_training=False):
        m = self.multiplier

        block1 = hk.Sequential(
            [hk.Conv2D(1 * m, 4, 2), ResBlock(1 * m), ResBlock(1 * m)],
            name="BackboneBlock1",
        )

        block2 = hk.Sequential(
            [hk.Conv2D(2 * m, 4, 2), ResBlock(2 * m), ResBlock(2 * m)],
            name="BackboneBlock2",
        )

        block3 = hk.Sequential(
            [hk.Conv2D(4 * m, 4, 2), ResBlock(4 * m), ResBlock(4 * m)],
            name="BackboneBlock2",
        )

        block4 = hk.Sequential(
            [
                hk.Conv2D(8 * m, 4, 2),
                ResBlock(8 * m),
                ResBlock(8 * m),
                ResBlock(8 * m),
                ResBlock(8 * m),
                ResBlock(8 * m),
            ],
            name="BackboneBlock2",
        )

        x1 = block1(x)
        x2 = block2(x1)
        x3 = block3(x2)
        x4 = block4(x3)
        return (x1, x2, x3, x4)


class UNetEncoder(hk.Module):
    def __init__(self, width=16, depth=6):
        super().__init__()
        self.width = width
        self.depth = depth

    def __call__(self, x, is_training=False):
        x = hk.Conv2D(self.width, 1)(x)

        # Contracting Path
        skip_connections = [x]
        for i in range(self.depth):
            x = unet.DownBlock(x, is_training)
            skip_connections.append(x)

        return skip_connections
