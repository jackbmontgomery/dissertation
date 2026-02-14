from abc import abstractmethod

from jax.lax import linalg
from jaxtyping import Array, PyTree, Scalar




# NOTE: This could be that c is some dim and the returns should be solution (t x dim) and current Float[Array, "t"]
class AbstractFDMSolver:
    @abstractmethod
    def compute_current(self, ck: Array) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def solve(self, params: PyTree) -> Scalar:
        raise NotImplementedError
