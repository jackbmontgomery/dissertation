from typing import Callable

import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import ECirreMechanismFDMParams
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver


class ECirreMechanismFDMSolver(AbstractFDMSolver):
    applied_potentials: Scalar
    dt: float
    X: Array
    num_x: int
    h: Scalar
    alpha_inner: Scalar
    sigma_inner: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h: float = 1e-2,
        dtheta: float = 5e-2,
    ):
        # Suggestion from Understanding Voltammetry 3.4.1
        dt = dtheta / voltammetry.sigma
        self.dt = dt

        T = jnp.linspace(
            voltammetry.t_min,
            voltammetry.t_max,
            int((voltammetry.t_max - voltammetry.t_min) / dt),
        )

        self.applied_potentials = vmap(voltammetry.applied_potential)(T)

        # Einstein on Brownian Motion
        x_max = 6.0 * jnp.sqrt(voltammetry.t_max)
        X = jnp.linspace(0.0, x_max, int(x_max / h))
        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")
        self.X = X
        self.num_x = len(X)

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, ck: Array) -> Array:
        c0 = ck[0]
        c1 = ck[2]
        c2 = ck[4]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcdx = (h2**2 * (c0 - c1) + h1**2 * (c2 - c0)) / (h1 * h2 * (h1 - h2))

        return -dcdx

    def _create_stepper(
        self,
        params: ECirreMechanismFDMParams,
    ) -> Callable[[Array, Scalar], Array]:
        def stepper(c_prev: Array, applied_potential: Scalar):
            k_red = params.K0 * jnp.exp(-params.alpha * (applied_potential - params.E0))
            k_ox = params.K0 * jnp.exp(
                (1.0 - params.alpha) * (applied_potential - params.E0)
            )

            # Second Lower Diagonal: Coefficient for the same species on the negative side
            d2l_inner = interleave_concat(
                self.alpha_inner, params.dB * self.alpha_inner
            )

            d2l = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),  # compatibility
                    d2l_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            # Lower Diagonal: Coefficients for A
            dlB_inner = jnp.zeros((self.num_x - 2,))
            dlA_inner = jnp.full((self.num_x - 2,), -1 * params.Kminus * self.dt)
            dl_inner = interleave_concat(dlB_inner, dlA_inner)

            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    jnp.array([-self.h * k_red / params.dB]),
                    dl_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            # Main Diagonal: beta_{n,A} -> beta_{0,A}, beta_{0,B} -> beta_{n,B}

            dA_inner = (
                1 - (self.alpha_inner + self.sigma_inner) + params.Kminus * self.dt
            )

            dB_inner = (
                1
                - params.dB * (self.alpha_inner + self.sigma_inner)
                + params.Kplus * self.dt
            )

            d_inner = interleave_concat(dA_inner, dB_inner)

            d = jnp.concat(
                [
                    jnp.array([1.0 + self.h * k_red, 1.0 + self.h * k_ox / params.dB]),
                    d_inner,
                    jnp.array([1.0, 1.0]),
                ]
            )

            # Upper Diagonal: alpha_{n-1, A} -> alpha_{0, A}, sigma_{0, B} -> sigma_{n-2, B}

            duB_inner = jnp.full((self.num_x - 2,), -1 * params.Kplus * self.dt)
            duA_inner = jnp.zeros((self.num_x - 2,))
            du_inner = interleave_concat(duB_inner, duA_inner)

            du = jnp.concat(
                [
                    jnp.array([-self.h * k_ox, 0.0]),
                    du_inner,
                    jnp.array([0.0]),
                    jnp.array([0.0]),  # compatibility
                ]
            )

            d2u_inner = interleave_concat(
                self.sigma_inner,
                params.dB * self.sigma_inner,
            )

            d2u = jnp.concatenate(
                [jnp.array([-1.0, -1.0]), d2u_inner, jnp.array([0.0, 0.0])]
            )  # compatibility

            rhs = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    jnp.array([0.0]),
                    c_prev[2:-2],
                    jnp.array([1.0]),
                    jnp.array([0.0]),
                ]
            )

            ck = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            current = self.compute_current(ck)

            return ck, current

        return stepper

    def solve(self, params: ECirreMechanismFDMParams) -> Scalar:
        # [A_0, B_0, A_1, B_1, ..., A_{N-1}, B_{N-1}]
        c_init = interleave_concat(jnp.ones_like(self.X), jnp.zeros_like(self.X))
        stepper = self._create_stepper(params)
        _, current = scan(stepper, c_init, self.applied_potentials)
        return current
