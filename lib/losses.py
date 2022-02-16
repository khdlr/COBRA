import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import haiku as hk

from jax.experimental.host_callback import id_print

from abc import ABC, abstractmethod

from .utils import pad_inf, fmt, distance_matrix, min_pool, draw_poly
from .jump_flood import jump_flood
from einops import rearrange


def stepwise(loss_fn):
  """Decorator that inserts each of `snake_steps` for `snake`
  and calls the wrapped loss_fn on it.
  Returns the mean loss accumulated in the process.
  """
  def inner(terms):
    losses = []
    for step in range(terms['snake_steps'].shape[1]):
      step_terms = {**terms, 'snake': terms['snake_steps'][:, step]}
      losses.append(loss_fn(step_terms))
    return jnp.mean(jnp.stack(losses))
  return inner


def l2_loss(terms):
  snake   = terms['snake']
  contour = terms['contour']
  loss = jnp.sum(jnp.square(snake - contour), axis=-1)
  loss = jnp.mean(loss)
  return loss


def l1_loss(terms):
  snake   = terms['snake']
  contour = terms['contour']
  loss = jnp.sum(jnp.abs(snake - contour), axis=-1)
  return loss


def min_min_loss(terms):
  snake   = terms['snake']
  contour = terms['contour']
  D = distance_matrix(snake, contour)
  min1 = D.min(axis=0)
  min2 = D.min(axis=1)
  min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
  return min_min


def offset_field_loss(terms):
  mask = terms['mask']
  offsets = terms['offsets']
  H, W = mask.shape
  true_offsets = jump_flood(mask)
  true_offsets = (true_offsets * (2/H))

  offsets = jax.image.resize(offsets, true_offsets.shape, 'bilinear')
  error = jnp.sum(jnp.square(offsets - true_offsets), axis=-1)

  return jnp.mean(error)


def bce(terms):
  seg  = terms['segmentation']
  mask = terms['mask']
  assert seg.shape == mask.shape
  return _bce(seg, mask)


def calfin_loss(terms):
  seg  = terms['segmentation']
  edge = terms['edge']
  mask = terms['mask']

  true_edge = hk.max_pool(mask, [5, 5], [1, 1], "SAME") != min_pool(mask, [5, 5], [1, 1], "SAME")

  seg_loss  = jnp.mean(_bce(seg, mask)) + \
              jnp.mean(_iou_loss(seg, mask))
  edge_loss = jnp.mean(_bce(edge, true_edge)) + \
              jnp.mean(_iou_loss(edge, true_edge))

  return (1/26) * seg_loss + (25/26) * edge_loss


def hed_unet_loss(terms):
  mask = terms['mask']
  prediction = terms['HED-UNet-Stack']

  edge = hk.max_pool(mask, [3, 3], [1, 1], "SAME") != min_pool(mask, [3, 3], [1, 1], "SAME")
  target = rearrange(jnp.stack([mask, edge], axis=-1), 'H W C -> H W 1 C', C=2)
  pred   = rearrange(prediction, 'H W (R C) -> H W R C', C=2)

  return _balanced_bce(pred, target)


class AbstractDTW(ABC):
  def __init__(self, bandwidth=None):
    self.bandwidth = bandwidth

  @abstractmethod
  def minimum(self, *args):
    pass

  def __call__(self, terms):
    snake   = terms['snake']
    contour = terms['contour']
    return self.dtw(snake, contour)

  def dtw(self, snake, contour):
      D = distance_matrix(snake, contour)
      # wlog: H >= W
      if D.shape[0] < D.shape[1]:
        D = D.T
      H, W = D.shape

      if self.bandwidth is not None:
        i, j = jnp.mgrid[0:H, 0:W]
        D = jnp.where(jnp.abs(i - j) > self.bandwidth,
          jnp.inf,
          D
        )

      y, x = jnp.mgrid[0:W+H-1, 0:H]
      indices = y - x
      model_matrix = jnp.where((indices < 0) | (indices >= W),
        jnp.inf,
        jnp.take_along_axis(D.T, indices, axis=0)
      )

      init = (
        pad_inf(model_matrix[0], 1, 0),
        pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
      )

      def scan_step(carry, current_antidiagonal):
        two_ago, one_ago = carry

        diagonal = two_ago[:-1]
        right    = one_ago[:-1]
        down     = one_ago[1:]
        best     = self.minimum(jnp.stack([diagonal, right, down], axis=-1))

        next_row = best + current_antidiagonal
        next_row = pad_inf(next_row, 1, 0)

        return (one_ago, next_row), next_row

      carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
      return carry[1][-1]


class DTW(AbstractDTW):
  __name__ = 'DTW'

  def minimum(self, args):
    return jnp.min(args, axis=-1)


def make_softmin(gamma, custom_grad=True):
  """
  We need to manually define the gradient of softmin
  to ensure (1) numerical stability and (2) prevent nans from
  propagating over valid values.
  """
  def softmin_raw(array):
    return -gamma * logsumexp(array / -gamma, axis=-1)

  if not custom_grad:
    return softmin_raw

  softmin = jax.custom_vjp(softmin_raw)

  def softmin_fwd(array):
    return softmin(array), (array / -gamma, )

  def softmin_bwd(res, g):
    scaled_array, = res
    grad = jnp.where(jnp.isinf(scaled_array),
      jnp.zeros(scaled_array.shape),
      jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1)
    )
    return grad,

  softmin.defvjp(softmin_fwd, softmin_bwd)
  return softmin


class SoftDTW(AbstractDTW):
  """
  SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
  by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)
  """
  __name__ = 'SoftDTW'

  def __init__(self, gamma=1.0, bandwidth=None):
    super().__init__(bandwidth)
    assert gamma > 0, "Gamma needs to be positive."
    self.gamma = gamma
    self.__name__ = f'SoftDTW({self.gamma})'
    self.minimum_impl = make_softmin(gamma)

  def minimum(self, args):
    return self.minimum_impl(args)


def marcos_dsac_loss(terms):
  mask    = terms['mask']
  snake   = jax.lax.stop_gradient(terms['snake'])
  contour = terms['contour']

  mapE = terms['mapE'][..., 0]
  mapA = terms['mapA']
  mapB = terms['mapB'][..., 0]
  # kappa is left out

  H, W = mask.shape

  # Regularization
  lossE = 0.005 * jnp.square(mapE)
  lossA = 0.005 * jnp.square(mapA)
  lossB = 0.005 * jnp.square(mapB)

  der1    = jnp.gradient(snake, axis=0)
  der2    = jnp.gradient(der1, axis=0)
  der1    = jnp.linalg.norm(der1, axis=-1)
  der2    = jnp.linalg.norm(der2, axis=-1)

  der1_GT = jnp.gradient(contour, axis=0)
  der2_GT = jnp.gradient(der1_GT, axis=0)
  der1_GT = jnp.linalg.norm(der1_GT, axis=-1)
  der2_GT = jnp.linalg.norm(der2_GT, axis=-1)

  lossE += mapE * (draw_poly(contour, [H, W], 12) - draw_poly(snake, [H, W], 12))
  lossA += mapA * (jnp.mean(der1_GT) - jnp.mean(der1))
  lossB += mapB * (draw_poly(contour, [H, W], 12, der2_GT) - draw_poly(snake, [H, W], 12, der2))

  return jnp.sum(lossE) + jnp.sum(lossB) + jnp.sum(lossA)


def dance_loss(terms):
  TODO


# Internals...
def _bce(seg, mask):
  logp   = jax.nn.log_sigmoid(seg)
  log1mp = jax.nn.log_sigmoid(-seg)
  loss   = -logp * mask - (log1mp) * (1-mask)

  return jnp.mean(loss)


def _balanced_bce(prediction, mask):
    beta = jnp.mean(mask, axis=[0, 1], keepdims=True)
    logit1 = jax.nn.log_sigmoid( prediction)
    logit0 = jax.nn.log_sigmoid(-prediction)

    loss = -jnp.mean(
        (1-beta) * logit1 * mask +
           beta  * logit0 * (1-mask)
    )

    return loss


def _iou_loss(prediction, mask):
    """corresponds to -log(iou_score) in the CALFIN code"""
    eps = 1e-1
    prediction = jax.nn.sigmoid(prediction[..., 0])

    intersection = jnp.sum(prediction * mask, axis=[0, 1])
    union        = jnp.sum(jnp.maximum(prediction, mask), axis=[0, 1])
    score        = (intersection + eps) / (union + eps)
    return -jnp.log(score)
