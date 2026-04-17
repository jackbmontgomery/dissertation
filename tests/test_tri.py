import jax.numpy as jnp
from jax import block_until_ready, grad, jit
from jax.lax import linalg

from src.linear_solvers import tridiagonal_solve


@jit
def tridiagonal_solve_jax(a, b, c, d):
    return linalg.tridiagonal_solve(
        jnp.concat([jnp.zeros(1), a]), b, jnp.concat([c, jnp.zeros(1)]), d[:, None]
    ).flatten()


grad_jax = grad(lambda a, b, c, d: jnp.sum(tridiagonal_solve_jax(a, b, c, d)))


@jit
def tridiagonal_solve_custom(a, b, c, d):
    return tridiagonal_solve(a, b, c, d)


grad_custom = grad(lambda a, b, c, d: jnp.sum(tridiagonal_solve_custom(a, b, c, d)))

N = 500
lower = -1 * jnp.ones((N - 1,))
main = 4 * jnp.ones((N,))
upper = -1 * jnp.ones((N - 1,))

A = jnp.diag(main) + jnp.diag(lower, k=-1) + jnp.diag(upper, k=1)
rhs = jnp.arange(1.0, N + 1.0)
solve_solution = jnp.linalg.solve(A, rhs)

_ = block_until_ready(tridiagonal_solve_custom(lower, main, upper, rhs))
_ = block_until_ready(tridiagonal_solve_jax(lower, main, upper, rhs))
_ = block_until_ready(grad_jax(lower, main, upper, rhs))
_ = block_until_ready(grad_custom(lower, main, upper, rhs))


# %%

_ = block_until_ready(tridiagonal_solve_custom(lower, main, upper, rhs))
_ = block_until_ready(tridiagonal_solve_jax(lower, main, upper, rhs))

_ = block_until_ready(grad_custom(lower, main, upper, rhs))
_ = block_until_ready(grad_jax(lower, main, upper, rhs))
