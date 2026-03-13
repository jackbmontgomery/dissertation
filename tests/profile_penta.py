from time import perf_counter

import jax.numpy as jnp

from src.solvers import pentadiagonal_solve

N = 10

d2l = -1 * jnp.ones((N - 2,))
dl = -1 * jnp.ones((N - 1,))
d = 6 * jnp.ones((N,))
du = -1 * jnp.ones((N - 1,))
d2u = -1 * jnp.ones((N - 2,))

A = (
    jnp.diag(d)
    + jnp.diag(dl, k=-1)
    + jnp.diag(du, k=1)
    + jnp.diag(d2l, k=-2)
    + jnp.diag(d2u, k=2)
)

rhs = jnp.array([1.0, 4.0, 6.0, 8.0, 20.0, 27.0, 3.0, 11.0, 5.0, 9.0])

warmup = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

times = []
for i in range(10):
    start = perf_counter()
    solution = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)
    end = perf_counter()
    times.append((end - start) * 1000)
print(times)
