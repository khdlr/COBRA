import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial

def dist(y1, x1, y2, x2):
    dy = y1 - y2
    dx = x1 - x2
    return dx*dx + dy*dy


def argminwhere(array, condition, axis):
    return jnp.argmin(jnp.where(condition, array, jnp.inf), axis=axis)


def roll2(ary, dy, dx):
    return jnp.roll(jnp.roll(ary, dy, axis=-2), dx, axis=-1)


def substep(data, size):
    neighbors = jnp.stack([
        roll2(data, -size, -size),
        roll2(data,     0, -size),
        roll2(data,  size, -size),
        roll2(data, -size,     0),
        roll2(data,  size,     0),
        roll2(data, -size,  size),
        roll2(data,     0,  size),
        roll2(data,  size,  size),
    ], axis=1)

    O_m, O_y, O_x, O_py, O_px = data
    O_d = jnp.where(O_py == -1,
        jnp.inf,
        dist(O_y, O_x, O_py, O_px)
    )
    N_m, N_y, N_x, N_py, N_px = neighbors
    N_d = dist(O_y, O_x, N_y, N_x)

    best_neighbor = argminwhere(N_d, (N_m != O_m), axis=0)[jnp.newaxis]

    get_neighbor = partial(jnp.take_along_axis, indices=best_neighbor, axis=0)
    B_m, B_y, B_x, _, _ = jax.vmap(get_neighbor)(neighbors)[:, 0]
    B_d      = jnp.take_along_axis(N_d, best_neighbor, axis=0)[0]

    data = jnp.where((B_m != O_m) & (B_d < O_d),
        jnp.stack([O_m, O_y, O_x, B_y, B_x]),
        data
    )
    
    O_m, O_y, O_x, O_py, O_px = data
    O_d = jnp.where(O_py == -1,
        jnp.inf,
        dist(O_y, O_x, O_py, O_px)
    )
    
    dist2pointer = dist(O_y, O_x, N_py, N_px)
    best_pointer = argminwhere(dist2pointer, (O_m == N_m) & (N_py > -1), axis=0)[jnp.newaxis]
    get_pointer  = partial(jnp.take_along_axis, indices=best_pointer, axis=0)
    P_m, _, _, P_y, P_x = jax.vmap(get_pointer)(neighbors)[:, 0]
    P_d = jnp.take_along_axis(dist2pointer, best_pointer, axis=0)[0] 
    
    data = jnp.where((P_m == O_m) & (P_y != -1) & (P_d < O_d),
        jnp.stack([O_m, O_y, O_x, P_y, P_x]),
        data
    )
    return data, None


@jax.jit
def jump_flood(mask):
    H, W = mask.shape
    steps = [1, 1]
    while steps[-1] < H and steps[-1] < W:
        steps.append(2 * steps[-1])
    steps = jnp.array(steps, dtype=jnp.int32)

    y, x = jnp.mgrid[0:H, 0:W]
    data = jnp.stack([
        mask,
        y,
        x,
        -jnp.ones_like(mask, dtype=jnp.int32), # pointer_y
        -jnp.ones_like(mask, dtype=jnp.int32), # pointer_x
    ]).astype(jnp.int32)

    result, _ = jax.lax.scan(substep, data, steps, reverse=True)

    m, y, x, py, px = result
    return jnp.stack([py-y, px-x], axis=-1)
