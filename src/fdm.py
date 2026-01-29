from abc import abstractmethod
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from equinox import Module, filter_jit
from jax import jit
from jax.lax import linalg, scan
from jaxtyping import Array, Scalar

from src.common import compute_current
from src.pde_parameters import ButlerVolmerParameters


class AbstractFDMDiscretisation(Module):
    @abstractmethod
    def operator(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerParameters
    ) -> Tuple[Scalar, Scalar, Scalar]:
        raise NotImplementedError

    @abstractmethod
    def rhs(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerParameters
    ) -> Array:
        raise NotImplementedError

    def operator_and_rhs(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerParameters
    ) -> Tuple[Tuple[Scalar, Scalar, Scalar], Scalar]:
        A = self.operator(c_prev, theta, params)
        b = self.rhs(c_prev, theta, params)
        return A, b


class ButlerVolmerFDMDiscretisation1D(AbstractFDMDiscretisation):
    h: Scalar
    lambda_: Scalar

    def __init__(self, X: Array):
        self.h = X[1] - X[0]
        self.lambda_ = self.h / (X[1:-1] - X[:-2]) ** 2

    def operator(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerParameters
    ) -> Tuple[Scalar, Scalar, Scalar]:
        dl = jnp.concatenate(
            [
                jnp.array([0.0]),
                -self.lambda_,
                jnp.array([0.0]),
            ]
        )

        b0 = 1 + self.h * jnp.exp(-params[0] * theta) * jnp.pow(params[1], 10.0) * (
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

    def rhs(
        self, c_prev: Scalar, theta: Scalar, params: ButlerVolmerParameters
    ) -> Array:
        inner = c_prev[1:-1]
        d0 = (
            self.h
            * jnp.exp(-params[0] * theta)
            * jnp.pow(params[1], 10.0)
            * jnp.exp(theta)
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


# @NOTE: The only way I can think of making this faster is by making the params static
#       during the solve and then we could optimise the computation of the operator and
#       rhs. But even then that is not where the bottleneck is


@filter_jit
def fdm_implicit_solve(
    params: ButlerVolmerParameters,
    c_init: Scalar,
    pde_discretisation: AbstractFDMDiscretisation,
    dx: float,
    potentials: Scalar,
):
    def fdm_stepper(ck: Array, theta: Scalar):
        (dl, d, du), b = pde_discretisation.operator_and_rhs(ck, theta, params)

        ck = tridiagonal_solve(dl, d, du, b)

        current = compute_current(ck, dx)

        return ck, (ck, current)

    _, (fdm_solution, current) = scan(fdm_stepper, c_init, potentials)

    return fdm_solution, current
