import jax
import jax.numpy as jnp
import haiku as hk


def identity(x, *aux, **kwaux):
    return x


class ReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)


class LeakyReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.leaky_relu(x)


def channel_dropout(x, rate):
    if rate < 0 or rate >= 1:
        raise ValueError("rate must be in [0, 1).")

    if rate == 0.0:
        return x

    keep_rate = 1.0 - rate
    mask_shape = (x.shape[0], *((1,) * (x.ndim-2)), x.shape[-1])

    keep = jax.random.bernoulli(hk.next_rng_key(), keep_rate, shape=mask_shape)
    return keep * x / keep_rate


def sample_dropout(x, rate):
    if rate < 0 or rate >= 1:
        raise ValueError("rate must be in [0, 1).")

    if rate == 0.0:
        return x

    keep_rate = 1.0 - rate
    mask_shape = (x.shape[0], *((1,) * (x.ndim-1)))

    keep = jax.random.bernoulli(hk.next_rng_key(), keep_rate, shape=mask_shape)
    return keep * x / keep_rate


class ConvBNAct(hk.Module):
    def __init__(self, *args, bn=True, act='relu', **kwargs):
        super().__init__()
        kwargs['with_bias'] = False
        self.conv = hk.Conv2D(*args, **kwargs)

        if bn:
            self.bn = hk.BatchNorm(True, True, 0.999)
        else:
            self.bn = identity

        if act is None:
            self.act = identity
        elif hasattr(jax.nn, act):
            self.act = getattr(jax.nn, act)
        else:
            raise ValueError(f"no activation called {act}")

    def __call__(self, x, is_training=False):
        x = self.conv(x)
        x = self.bn(x, is_training)
        x = self.act(x)
        return x


class SepConvBN(hk.Module):
    def __init__(self, filters, stride=1, kernel_size=3, rate=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.filters = filters

    def __call__(self, x, is_training=False):
        x = hk.Conv2D(self.filters, 1)(x)
        x = ConvBNAct(self.filters, self.kernel_size, stride=self.stride,
            rate=self.rate, feature_group_count=self.filters)(x, is_training)

        return x


class ConvBNAct1D(hk.Module):
    def __init__(self, *args, bn=True, act='relu', **kwargs):
        super().__init__()
        kwargs['with_bias'] = False
        self.conv = hk.Conv1D(*args, **kwargs)

        if bn:
            self.bn = hk.BatchNorm(True, True, 0.999)
        else:
            self.bn = identity

        if act is None:
            self.act = identity
        elif hasattr(jax.nn, act):
            self.act = getattr(jax.nn, act)
        else:
            raise ValueError(f"no activation called {act}")

    def __call__(self, x, is_training=False):
        x = self.conv(x)
        x = self.bn(x, is_training)
        x = self.act(x)
        return x


class SepConvBN1D(hk.Module):
    def __init__(self, filters, stride=1, kernel_size=3, rate=1, depth_activation=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.filters = filters
        self.act = 'relu' if depth_activation else None

    def __call__(self, x, is_training=False):
        B, T, C = x.shape
        if self.act is None:
            x = jax.nn.relu(x)

        x = ConvBNAct1D(C, self.kernel_size, stride=self.stride,
            rate=self.rate, feature_group_count=C, act=self.act, bn=False)(x, is_training)
        x = ConvBNAct1D(self.filters, 1, bn=False)(x, is_training)

        return x


def upsample(x, factor=None, shp=None):
    B, H, W, C = x.shape
    if factor is not None:
        H *= factor
        W *= factor
    else:
        H, W = shp
    return jax.image.resize(x, [B, H, W, C], 'bilinear')


