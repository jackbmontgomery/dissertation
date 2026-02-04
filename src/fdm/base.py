from abc import abstractmethod

from jax.lax import linalg
from jaxtyping import Array, Scalar

from src.params import MacroElectrodeParams


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return linalg.tridiagonal_solve(dl, d, du, b[:, None]).flatten()


# NOTE: This could be that c is some dim and the returns should be solution (t x dim) and current Float[Array, "t"]
class AbstractFDMSolver:
    @abstractmethod
    def compute_current(self, ck: Array) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def solve(self, params: MacroElectrodeParams) -> Scalar:
        raise NotImplementedError
