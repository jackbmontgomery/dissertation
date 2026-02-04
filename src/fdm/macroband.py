from typing import Callable

import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import ElectrodeKineticsParameters
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver, tridiagonal_solve


class MacroElectrodeFDMSolver(AbstractFDMSolver):
    applied_potentials: Scalar
    X: Array
    h: Scalar
    dl_inner: Scalar
    d_inner: Scalar
    du_inner: Scalar

    def __init__(
        self, voltammetry: AbstractVoltammetryTechnique, h: float, dtheta: float
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
        X = jnp.linspace(0.0, x_max, int(x_max / h))
        print("Space Discretisation", f"X: {X.shape}", f"T: {T.shape}")
        self.X = X

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.dl_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.du_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))
        self.d_inner = 1 - self.dl_inner - self.du_inner

    def compute_current(self, ck: Array) -> Array:
        c0 = ck[0]
        c1 = ck[1]
        c2 = ck[2]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcdx = (h2**2 * (c0 - c1) + h1**2 * (c2 - c0)) / (h1 * h2 * (h1 - h2))

        return -dcdx

    def _create_stepper(
        self,
        params: ElectrodeKineticsParameters,
    ) -> Callable[[Array, Scalar], Array]:
        def stepper(c_prev: Array, applied_potential: Scalar):
            dl = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    self.dl_inner,
                    jnp.array([0.0]),
                ]
            )

            d0 = 1 + self.h * jnp.exp(
                -params.alpha * (applied_potential - params.epsilon)
            ) * params.kappa * (1 + jnp.exp(applied_potential - params.epsilon))
            d = jnp.concatenate(
                [
                    jnp.array([d0]),
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

            rhs0 = (
                self.h
                * jnp.exp(-params.alpha * (applied_potential - params.epsilon))
                * params.kappa
                * jnp.exp(applied_potential - params.epsilon)
            )

            rhs = jnp.concatenate(
                [
                    jnp.array([rhs0]),
                    c_prev[1:-1],
                    jnp.array([1.0]),
                ]
            )

            ck = tridiagonal_solve(dl, d, du, rhs)

            current = self.compute_current(ck)

            return ck, current

        return stepper

    def solve(self, params: ElectrodeKineticsParameters) -> Scalar:
        c_init = jnp.ones_like(self.X)
        stepper = self._create_stepper(params)
        _, current = scan(stepper, c_init, self.applied_potentials)
        return current
