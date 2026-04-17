import jax.numpy as jnp
from jax import block_until_ready

from src.linear_solvers import tridiagonal_solve

N = 10
lower = -1 * jnp.ones((N - 1,))
main = 4 * jnp.ones((N,))
upper = -1 * jnp.ones((N - 1,))

A = jnp.diag(main) + jnp.diag(lower, k=-1) + jnp.diag(upper, k=1)
rhs = jnp.arange(1.0, N + 1.0)
solve_solution = jnp.linalg.solve(A, rhs)

tri_solution = block_until_ready(tridiagonal_solve(lower, main, upper, rhs))


# %%
# 83.7 μs ± 443 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
tri_solution = block_until_ready(tridiagonal_solve(lower, main, upper, rhs))
