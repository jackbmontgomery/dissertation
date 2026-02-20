from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import PyTree, Scalar


class AbstractFDMSolver:
    @abstractmethod
    def compute_current(self, c: PyTree) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def solve(self, params: PyTree) -> Scalar:
        raise NotImplementedError


def uniform_discretisation(max_val: Scalar, h: float):
    return jnp.linspace(0.0, max_val, int(max_val / h))


def exponential_discretisation(max_val: Scalar, h: float, omega: float):
    X = [0.0]
    while X[-1] < max_val:
        X.append(X[-1] + h)
        h *= omega
    return jnp.array(X)
