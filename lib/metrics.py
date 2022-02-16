import jax
import jax.numpy as jnp


def mae(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.mean(jnp.sqrt(squared_distances))


def rmse(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.sqrt(jnp.mean(squared_distances))


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


def forward_mae(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(terms):
  snake   = terms['snake']
  contour = terms['contour']
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.sqrt(jnp.mean(squared_dist))


def symmetric_mae(terms):
  return 0.5 * forward_mae(terms) + 0.5 * backward_mae(terms)


def symmetric_rmse(terms):
    return 0.5 * forward_rmse(terms) + 0.5 * backward_rmse(terms)


