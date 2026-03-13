import jax.numpy as jnp
from jax import custom_vjp, jit
from jax.lax import linalg
from jaxtyping import Scalar


@custom_vjp
def tridiagonal_solve(a, b, c, d):
    return tridiagonal_solve_impl(a, b, c, d)


def tridiagonal_solve_fwd(a, b, c, d):
    x = tridiagonal_solve_impl(a, b, c, d)
    return x, (a, b, c, x)


def tridiagonal_solve_bwd(res, g):
    a, b, c, x = res
    lam = tridiagonal_solve_impl(c, b, a, g)
    g_b = -lam * x
    g_a = -lam[1:] * x[:-1]
    g_c = -lam[:-1] * x[1:]
    g_d = lam
    return g_a, g_b, g_c, g_d


tridiagonal_solve.defvjp(tridiagonal_solve_fwd, tridiagonal_solve_bwd)


@jit
def tridiagonal_solve_impl(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> Scalar:
    return linalg.tridiagonal_solve(
        jnp.concat([jnp.zeros(1), a]), b, jnp.concat([c, jnp.zeros(1)]), d[:, None]
    ).flatten()
