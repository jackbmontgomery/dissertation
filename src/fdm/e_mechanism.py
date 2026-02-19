from typing import Callable

import jax.numpy as jnp
from jax import vmap
from jax.lax import pcast, scan
from jaxtyping import Array, Scalar

from src.params import EMechanismFDMParams
from src.solvers import tridiagonal_solve
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver, uniform_discretisation


class EMechanismFDMSolver(AbstractFDMSolver):
    applied_potentials: Scalar
    X: Array
    num_x: int
    h: Scalar
    alpha_inner: Scalar
    b_inner: Scalar
    sigma_inner: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h: float = 5e-3,
        dtheta: float = 5e-2,
    ):
        # Suggestion from Understanding Voltammetry 3.4.1
        dt = dtheta / voltammetry.sigma

        T = jnp.linspace(
            voltammetry.t_min,
            voltammetry.t_max,
            int((voltammetry.t_max - voltammetry.t_min) / dt),
        )

        self.applied_potentials = vmap(voltammetry.applied_potential)(T)

        # Einstein on Brownian Motion
        x_max = 6.0 * jnp.sqrt(voltammetry.t_max)
        X = uniform_discretisation(x_max, h)

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")
        self.X = X
        self.num_x = len(X)

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, ck: Array) -> Array:
        half_idx = self.num_x - 1
        c0 = ck[half_idx]
        c1 = ck[half_idx - 1]
        c2 = ck[half_idx - 2]

        dcdx = (-c2 + 4 * c1 - 3 * c0) / (2 * self.h)

        return -dcdx

    def _create_stepper(
        self,
        params: EMechanismFDMParams,
    ) -> Callable[[Array, Scalar], Array]:
        def stepper(c_prev: Array, applied_potential: Scalar):
            # Lower Diagonal: sigma{n-2,A} -> sigma_{0,A}, alpha_{0,B} -> alpha_{n-1,B}, 0.0

            alpha_B0 = (
                -self.h
                * params.K0
                * jnp.exp(-params.alpha * (applied_potential - params.E0))
                / params.dB
            )

            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    self.sigma_inner,
                    jnp.array([-1.0]),
                    jnp.array([alpha_B0]),
                    params.dB * self.alpha_inner,
                    jnp.array([0.0]),
                ]
            )

            # Main Diagonal: beta_{n,A} -> beta_{0,A}, beta_{0,B} -> beta_{n,B}
            beta_A0 = 1 + self.h * params.K0 * jnp.exp(
                -params.alpha * (applied_potential - params.E0)
            )
            beta_B0 = (
                1
                + self.h
                * params.K0
                * jnp.exp((1 - params.alpha) * (applied_potential - params.E0))
                / params.dB
            )

            d = jnp.concat(
                [
                    jnp.array([1.0]),
                    1 - (self.alpha_inner + self.sigma_inner),
                    jnp.array([beta_A0, beta_B0]),
                    1 - params.dB * (self.alpha_inner + self.sigma_inner),
                    jnp.array([1.0]),
                ]
            )

            alpha_A0 = (
                -self.h
                * params.K0
                * jnp.exp((1 - params.alpha) * (applied_potential - params.E0))
            )

            # Upper Diagonal: alpha_{n-1, A} -> alpha_{0, A}, sigma_{0, B} -> sigma_{n-2, B}
            du = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    self.alpha_inner,
                    jnp.array([alpha_A0]),
                    jnp.array([-1.0]),
                    params.dB * self.sigma_inner,
                    jnp.array([0.0]),  # compatibility
                ]
            )

            rhs = jnp.concat(
                [
                    jnp.array([1.0]),
                    c_prev[1 : self.num_x - 1],
                    jnp.array([0.0, 0.0]),
                    c_prev[self.num_x + 1 : -1],
                    jnp.array([0.0]),
                ]
            )

            ck = tridiagonal_solve(dl, d, du, rhs)

            current = self.compute_current(ck)

            return ck, current

        return stepper

    def solve(self, params: EMechanismFDMParams) -> Scalar:
        # [A_{N-1},..., A_{0}, B_{0},..., B_{N-1}]
        c_init = jnp.concat([jnp.ones_like(self.X), jnp.zeros_like(self.X)])
        stepper = self._create_stepper(params)
        _, current = scan(stepper, c_init, self.applied_potentials)
        return current
