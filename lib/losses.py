import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print
import haiku as hk
import optax

from abc import ABC, abstractmethod
from inspect import signature

from .utils import pad_inf, fmt, distance_matrix, min_pool, draw_poly
from .jump_flood import jump_flood
from einops import rearrange


def call_loss(loss_fn, terms):
  # TODO: Inspection
  sig = signature(loss_fn)
  args = {k: v for k, v in terms.items() if k in sig.parameters}

  loss_terms = jax.vmap(loss_fn)(**args)
  loss_terms = jax.tree_map(jnp.mean, loss_terms)
  if isinstance(loss_terms, dict):
    total_loss = sum(loss_terms.values())
  else:
    total_loss = loss_terms
    loss_terms = {}
  loss_terms['loss'] = total_loss
  return total_loss, loss_terms


def stepwise(loss_fn):
  """Decorator that inserts each of `snake_steps` for `snake`
  and calls the wrapped loss_fn on it.
  Returns the mean loss accumulated in the process.
  """
  def inner(snake_steps, contour):
    losses = {}
    for step in range(len(snake_steps)):
      step_terms = {'snake': snake_steps[step], 'contour': contour}
      losses[f'loss_{step}'] = loss_fn(**step_terms)
    return losses
  return inner


##### Generic Loss Functions ####


def l2_loss(snake, contour):
  loss = jnp.sum(jnp.square(snake - contour), axis=-1)
  loss = jnp.mean(loss)
  return loss


def l1_loss(snake, contour):
  loss = jnp.sum(jnp.abs(snake - contour), axis=-1)
  return loss


def huber_loss(snake, contour):
  loss = optax.huber_loss(snake, contour, delta=0.05)
  return loss


def min_min_loss(snake, contour):
  D = distance_matrix(snake, contour)
  min1 = D.min(axis=0)
  min2 = D.min(axis=1)
  min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
  return min_min


def offset_field_loss(offsets, mask):
  H, W = mask.shape
  true_offsets = jump_flood(mask)
  true_offsets = (true_offsets * (2/H))

  offsets = jax.image.resize(offsets, true_offsets.shape, 'bilinear')
  error = jnp.sum(jnp.square(offsets - true_offsets), axis=-1)

  return jnp.mean(error)


def bce(segmentation, mask):
  seg  = segmentation
  if len(seg.shape) == 3 and seg.shape[-1] == 1:
    seg = seg[..., 0]

  seg = jax.image.resize(seg, mask.shape, 'bilinear')
  assert seg.shape == mask.shape, f"{seg.shape} != {mask.shape}"
  return _bce(seg, mask)


##### METRICS #####


def mae(snake, contour):
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.mean(jnp.sqrt(squared_distances))


def rmse(snake, contour):
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.sqrt(jnp.mean(squared_distances))


def forward_mae(snake, contour):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(snake, contour):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(snake, contour):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(snake, contour):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.sqrt(jnp.mean(squared_dist))


def symmetric_mae(snake, contour):
  return 0.5 * forward_mae(snake, contour) + 0.5 * backward_mae(snake, contour)


def symmetric_rmse(snake, contour):
  return 0.5 * forward_rmse(snake, contour) + 0.5 * backward_rmse(snake, contour)


##### Architecture Specific Loss Functions #####


def stepwise_softdtw_and_aux(snake_steps, segmentation, offsets, mask, contour):
  loss_terms = stepwise(SoftDTW(0.001))(snake_steps, contour)
  loss_terms['segmentation_loss'] = bce(segmentation, mask)
  loss_terms['offset_loss'] = offset_field_loss(offsets, mask)

  return loss_terms


def calfin_loss(segmentation, edge, mask):
  seg  = segmentation

  true_edge = hk.max_pool(mask, [5, 5], [1, 1], "SAME") != min_pool(mask, [5, 5], [1, 1], "SAME")

  seg_loss  = jnp.mean(_bce(seg, mask)) + \
              jnp.mean(_iou_loss(seg, mask))
  edge_loss = jnp.mean(_bce(edge, true_edge)) + \
              jnp.mean(_iou_loss(edge, true_edge))

  return (1/26) * seg_loss + (25/26) * edge_loss


def hed_unet_loss(hed_unet_stack, mask):
  prediction = hed_unet_stack

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

  def __call__(self, snake, contour):
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


def marcos_dsac_loss(snake,  mapE, mapA, mapB, mask, contour):
  mapE = mapE[..., 0]
  mapA = mapA
  mapB = mapB[..., 0]
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


def dance_loss(snake_steps, edge, mask):
  edge = edge[..., 0]

  loss_terms = {}

  # Edge loss
  true_edge = hk.max_pool(mask, [3, 3], [1, 1], "SAME") != min_pool(mask, [3, 3], [1, 1], "SAME")
  edge = jax.image.resize(edge, true_edge.shape, 'bilinear')

  p2 = jnp.sum(edge * edge)
  g2 = jnp.sum(true_edge * true_edge)
  pg = jnp.sum(edge * true_edge)

  dice_coef = (2*pg) / (p2+g2+0.0001)
  loss_terms['dice_loss'] = 1.0 - dice_coef

  weights = jax.nn.softmax(jnp.array([1, 1, 2]) / 3)
  for i, (weight, snake) in enumerate(zip(weights, snake_steps[1:])):
    loss_terms[f'stage_{i}_loss'] = 10 * optax.huber_loss(snake, contour, delta=0.033)

  return loss_terms


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



def squared_distance_point_to_linesegment(point, linestart, lineend):
    p = point
    b = lineend
    a = linestart
    
    b_a = b - a
    p_a = p - a

    t = jnp.dot(b_a, p_a) / jnp.dot(b_a, b_a)
    t = jnp.nan_to_num(jnp.clip(t, 0, 1), nan=0.0, posinf=0.0, neginf=0.0)
    
    dist2 = jnp.sum(jnp.square((1-t)*a + t*b - p))
    
    return dist2


def squared_distance_point_to_curve(point, polyline):
    startpoints = polyline[:-1]
    endpoints   = polyline[1:]
    
    get_squared_distances = jax.vmap(squared_distance_point_to_linesegment,
                                 in_axes=[None, 0, 0])
    squared_distances = get_squared_distances(point, startpoints, endpoints)
    
    min_dist = jnp.nanmin(squared_distances)
    return jnp.where(jnp.isnan(min_dist), 0, min_dist)


def squared_distance_points_to_curve(points, polyline):
    get_point_to_curve = jax.vmap(squared_distance_point_to_curve,
                                 in_axes=[0, None])
    return get_point_to_curve(points, polyline)

