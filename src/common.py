from jax import Array


def compute_current(c: Array, dx: float) -> Array:
    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    return -(-c2 + 4.0 * c1 - 3.0 * c0) / (2.0 * dx)
