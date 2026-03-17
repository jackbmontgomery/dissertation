from abc import abstractmethod
from typing import Any, Tuple

import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from src.voltammetry import AbstractVoltammetryTechnique


class AbstractFDSolver:
    @abstractmethod
    def compute_current(self, *args: Any, **kwargs: Any) -> Scalar:
        raise NotImplementedError

    @abstractmethod
    def solve(self, params: PyTree) -> Tuple[Array, Scalar]:
        raise NotImplementedError


def uniform_discretisation(max_val: Scalar, h: float):
    return jnp.linspace(0.0, max_val, int(max_val / h))


def exponential_discretisation(max_val: Scalar, h: float, omega: float):
    X = [0.0]
    while X[-1] < max_val:
        X.append(X[-1] + h)
        h *= omega
    return jnp.array(X)


def setup_fd_discritisation(
    voltammetry: AbstractVoltammetryTechnique, dtheta: float, h0: float, omega: float
):
    dt = dtheta / voltammetry.sigma

    T = jnp.linspace(
        voltammetry.t_min,
        voltammetry.t_max,
        int((voltammetry.t_max - voltammetry.t_min) / dt),
    )

    # Einstein on Brownian Motion
    x_max = 6.0 * jnp.sqrt(voltammetry.t_max)
    X = exponential_discretisation(x_max, h0, omega)

    print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")

    X_plus = X[2:] - X[1:-1]
    X_minus = X[1:-1] - X[:-2]

    alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
    gamma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))
    return T, dt, X, alpha_inner, gamma_inner
