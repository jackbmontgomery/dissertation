from jax.lax import linalg
from jaxtyping import Scalar


def tridiagonal_solve(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> Scalar:
    return linalg.tridiagonal_solve(a, b, c, d[:, None]).flatten()
