from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import EMechanismFDMParams
from src.solvers import tridiagonal_solve
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver, uniform_discretisation


@dataclass
class ScanInputSequence:
    alpha_A0: Scalar
    alpha_B0: Scalar
    beta_A0: Scalar
    beta_B0: Scalar
    K_red: Scalar
    K_ox: Scalar


class EMechanismFDMSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Array
    Nx: int
    h1: Scalar
    hs: Scalar
    alpha_inner: Scalar
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
        self.Nx = len(X)

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h1 = X[1] - X[0]
        self.h2 = X[2] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, c: Scalar, K_red: Scalar, K_ox: Scalar) -> Scalar:
        c0_A = c[self.Nx - 1]
        c0_B = c[self.Nx]
        return -(K_red * c0_A - K_ox * c0_B)

    def _create_stepper(
        self,
        params: EMechanismFDMParams,
    ) -> Callable[
        [Scalar, ScanInputSequence],
        Tuple[Scalar, Scalar],
    ]:
        def stepper(c_prev: Scalar, x: ScanInputSequence) -> Tuple[Scalar, Scalar]:
            # Lower Diagonal: sigma{n-2,A} -> sigma_{0,A}, alpha_{0,B} -> alpha_{n-1,B}, 0.0

            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    self.sigma_inner,
                    jnp.array([-1.0, x.alpha_B0]),
                    params.dB * self.alpha_inner,
                    jnp.array([0.0]),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([1.0]),
                    1 - (self.alpha_inner + self.sigma_inner),
                    jnp.array([x.beta_A0, x.beta_B0]),
                    1 - params.dB * (self.alpha_inner + self.sigma_inner),
                    jnp.array([1.0]),
                ]
            )

            du = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    self.alpha_inner,
                    jnp.array([x.alpha_A0]),
                    jnp.array([-1.0]),
                    params.dB * self.sigma_inner,
                    jnp.array([0.0]),  # compatibility
                ]
            )

            rhs = jnp.concat(
                [
                    jnp.array([1.0]),
                    c_prev[1 : self.Nx - 1],
                    jnp.array([0.0, 0.0]),
                    c_prev[self.Nx + 1 : -1],
                    jnp.array([0.0]),
                ]
            )

            Ck = tridiagonal_solve(dl, d, du, rhs)

            current = self.compute_current(Ck, x.K_red, x.K_ox)

            return Ck, current

        return stepper

    def solve(self, params: EMechanismFDMParams) -> Scalar:
        stepper = self._create_stepper(params)

        c_init = jnp.concat([jnp.ones_like(self.X), jnp.zeros_like(self.X)])

        K_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.E0)
        )

        K_ox = params.K0 * jnp.exp(
            (1 - params.alpha) * (self.applied_potentials - params.E0)
        )

        alpha_A0 = -self.h1 * K_ox
        alpha_B0 = -self.h1 * K_red / params.dB

        beta_A0 = 1 + self.h1 * K_red
        beta_B0 = 1 + self.h1 * K_ox / params.dB

        xs = ScanInputSequence(
            alpha_A0=alpha_A0,
            alpha_B0=alpha_B0,
            beta_A0=beta_A0,
            beta_B0=beta_B0,
            K_red=K_red,
            K_ox=K_ox,
        )

        _, current = scan(stepper, c_init, xs)
        return current
