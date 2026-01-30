from abc import abstractmethod
from typing import Tuple

import jax.numpy as jnp
from equinox import Module, filter_jit
from jax.lax import linalg, scan
from jaxtyping import Array, Scalar

from src.experiment import CyclicMacroBand1D, LinearSweepMacroBand
from src.pde_parameters import AbstractPDEParameters, ButlerVolmerPhysicalParameters


def discretise_experiment(
    experiment: CyclicMacroBand1D | LinearSweepMacroBand,
    dx: float = 1e-3,
    dtheta: float = 0.05,
) -> Tuple[Scalar, Scalar]:
    dt = dtheta / experiment.sigma

    T = jnp.linspace(
        experiment.t_min,
        experiment.t_max,
        int((experiment.t_max - experiment.t_min) / dt),
    )

    X = jnp.linspace(
        experiment.x_min,
        experiment.x_max,
        int((experiment.x_max - experiment.x_min) / dx),
    )

    return T, X


class AbstractFDMDiscretisation(Module):
    @abstractmethod
    def operator(
        self, c_prev: Scalar, theta: Scalar, params: AbstractPDEParameters
    ) -> Tuple[Scalar, Scalar, Scalar]:
        raise NotImplementedError

    @abstractmethod
    def vector(
        self, c_prev: Scalar, theta: Scalar, params: AbstractPDEParameters
    ) -> Scalar:
        raise NotImplementedError

    def operator_and_vector(
        self, c_prev: Scalar, theta: Scalar, params: AbstractPDEParameters
    ) -> Tuple[Tuple[Scalar, Scalar, Scalar], Scalar]:
        A = self.operator(c_prev, theta, params)
        b = self.vector(c_prev, theta, params)
        return A, b


class ButlerVolmerFDMDiscretisation1D(AbstractFDMDiscretisation):
    h: Scalar
    lambda_: Scalar

    def __init__(self, X: Array):
        self.h = X[1] - X[0]
        self.lambda_ = self.h / (X[1:-1] - X[:-2]) ** 2

    def operator(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerPhysicalParameters
    ) -> Tuple[Scalar, Scalar, Scalar]:  # ty: ignore[invalid-method-override]
        dl = jnp.concatenate(
            [
                jnp.array([0.0]),
                -self.lambda_,
                jnp.array([0.0]),
            ]
        )

        b0 = 1 + self.h * jnp.exp(-params.alpha * theta) * params.kappa0 * (
            1 + jnp.exp(theta)
        )
        d = jnp.concatenate(
            [
                jnp.array([b0]),
                1 + 2 * self.lambda_,
                jnp.array([1.0]),
            ],
        )

        du = jnp.concatenate(
            [
                jnp.array([-1.0]),
                -self.lambda_,
                jnp.array([0.0]),
            ]
        )

        return dl, d, du

    def vector(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerPhysicalParameters
    ) -> Array:  # ty: ignore[invalid-method-override]
        inner = c_prev[1:-1]
        d0 = self.h * jnp.exp(-params.alpha * theta) * params.kappa0 * jnp.exp(theta)
        return jnp.concatenate(
            [
                jnp.array([d0]),
                inner,
                jnp.array([1.0]),
            ]
        )


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return linalg.tridiagonal_solve(dl, d, du, b[:, None]).flatten()


@filter_jit
def fdm_implicit_solve(
    init: Scalar,
    forcing: Scalar,
    fdm_discretisation: AbstractFDMDiscretisation,
    fdm_params: AbstractPDEParameters,
):
    def fdm_stepper(carry: Array, forcing: Scalar):
        (dl, d, du), b = fdm_discretisation.operator_and_vector(
            carry, forcing, fdm_params
        )

        carry = tridiagonal_solve(dl, d, du, b)

        return carry, carry

    _, fdm_solution = scan(fdm_stepper, init, forcing)

    return fdm_solution
