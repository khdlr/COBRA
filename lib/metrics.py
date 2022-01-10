import jax
import jax.numpy as jnp


def mae(prediction, snake):
    squared_distances = jnp.sum(jnp.square(prediction - snake), axis=-1)
    return jnp.mean(jnp.sqrt(squared_distances))


def rmse(prediction, snake):
    squared_distances = jnp.sum(jnp.square(prediction - snake), axis=-1)
    return jnp.sqrt(jnp.mean(squared_distances))


def squared_distance_point_to_linesegment(point, linestart, lineend):
    p = point
    b = lineend
    a = linestart
    
    b_a = b - a
    p_a = p - a

    t = jnp.dot(b_a, p_a) / jnp.dot(b_a, b_a)
    t = jnp.clip(t, 0, 1)
    
    dist2 = jnp.sum(jnp.square((1-t)*a + t*b - p))
    
    return dist2


def squared_distance_point_to_best_segment(point, polyline):
    startpoints = polyline[:-1]
    endpoints   = polyline[1:]
    
    get_squared_distances = jax.vmap(squared_distance_point_to_linesegment,
                                 in_axes=[None, 0, 0])
    squared_distances = get_squared_distances(point, startpoints, endpoints)
    
    min_dist = jnp.nanmin(squared_distances)
    return jnp.where(jnp.isnan(min_dist), 0, min_dist)


def squared_distance_points_to_best_segment(points, polyline):
    get_point_to_best_segment = jax.vmap(squared_distance_point_to_best_segment,
                                 in_axes=[0, None])
    return get_point_to_best_segment(points, polyline)


def mae_line_to_line(line1, line2):
    """
    Calculates the avg deviation of vertices in line1
    to line2
    """
    squared_dist = squared_distance_points_to_best_segment(line1, line2)
    return jnp.mean(jnp.sqrt(squared_dist))
    

def rmse_line_to_line(line1, line2):
    """
    Calculates the rmse of 
    """
    squared_dist = squared_distance_points_to_best_segment(line1, line2)
    return jnp.sqrt(jnp.mean(squared_dist))


def forward_mae(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(prediction, snake)
    return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(snake, prediction)
    return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(prediction, snake)
    return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(prediction, snake):
    squared_dist = squared_distance_points_to_best_segment(snake, prediction)
    return jnp.sqrt(jnp.mean(squared_dist))


def symmetric_mae(prediction, snake):
    return 0.5 * forward_mae(prediction, snake) + 0.5 * backward_mae(prediction, snake)


def symmetric_rmse(prediction, snake):
    return 0.5 * forward_rmse(prediction, snake) + 0.5 * backward_rmse(prediction, snake)


