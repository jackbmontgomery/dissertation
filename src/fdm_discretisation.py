from abc import abstractmethod
from typing import Tuple

import jax.numpy as jnp
from equinox import Module, filter_jit
from jax.lax import linalg, scan
from jaxtyping import Array, Scalar

from src.experiment import (
    AbstractExperiment,
)
from src.pde_parameters import AbstractPDEParameters, ButlerVolmerPhysicalParameters


def exponentially_expanding_discretise(
    experiment: AbstractExperiment,
    h: float = 1e-2,
    dtheta: float = 0.05,
    omega: float = 1.01,
) -> Tuple[Scalar, Scalar]:
    dt = dtheta / experiment.sigma

    T = jnp.linspace(
        experiment.t_min,
        experiment.t_max,
        int((experiment.t_max - experiment.t_min) / dt),
    )

    X = [0.0]
    while X[-1] < experiment.x_max:
        X.append(X[-1] + h)
        h *= omega

    X[-1] = experiment.x_max

    X = jnp.array(X)

    return T, X


def uniform_discretise(
    experiment: AbstractExperiment,
    dx: float = 1e-2,
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
    dl_inner: Scalar
    d_inner: Scalar
    du_inner: Scalar

    def __init__(self, X: Array, T: Array):
        dt = T[1] - T[0]
        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.dl_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.du_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))
        self.d_inner = 1 - self.dl_inner - self.du_inner

    def operator(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerPhysicalParameters
    ) -> Tuple[Scalar, Scalar, Scalar]:  # ty: ignore[invalid-method-override]
        dl = jnp.concatenate(
            [
                jnp.array([0.0]),
                self.dl_inner,
                jnp.array([0.0]),
            ]
        )

        b0 = 1 + self.h * jnp.exp(
            -params.alpha * (theta - params.eps0)
        ) * params.kappa0 * (1 + jnp.exp(theta - params.eps0))
        d = jnp.concatenate(
            [
                jnp.array([b0]),
                self.d_inner,
                jnp.array([1.0]),
            ],
        )

        du = jnp.concatenate(
            [
                jnp.array([-1.0]),
                self.du_inner,
                jnp.array([0.0]),
            ]
        )

        return dl, d, du

    def vector(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerPhysicalParameters
    ) -> Array:  # ty: ignore[invalid-method-override]
        inner = c_prev[1:-1]
        d0 = (
            self.h
            * jnp.exp(-params.alpha * (theta - params.eps0))
            * params.kappa0
            * jnp.exp(theta - params.eps0)
        )
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
