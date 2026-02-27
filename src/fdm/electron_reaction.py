from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import ElectronReactionParams
from src.solvers import tridiagonal_solve
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver, exponential_discretisation


@dataclass
class ScanInputSequence:
    alpha_A0: Scalar
    alpha_B0: Scalar
    beta_A0: Scalar
    beta_B0: Scalar
    K_red: Scalar
    K_ox: Scalar


class ElectronReactionFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Array
    Nx: int
    h0: Scalar
    hs: Scalar
    alpha_inner: Scalar
    sigma_inner: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h: float = 1e-3,
        dtheta: float = 1e-1,
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
        X = exponential_discretisation(x_max, h, 1.1)

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")
        self.X = X
        self.Nx = len(X)

        X_plus = X[2:] - X[1:-1]
        X_minus = X[1:-1] - X[:-2]

        self.h0 = X[1] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, c: Scalar, K_red: Scalar, K_ox: Scalar) -> Scalar:
        c0_A = c[self.Nx - 1]
        c0_B = c[self.Nx]
        return -(K_red * c0_A - K_ox * c0_B)

    def create_stepper(
        self,
        params: ElectronReactionParams,
    ) -> Callable[
        [Scalar, ScanInputSequence],
        Tuple[Scalar, Scalar],
    ]:
        dl_inner_A = jnp.flip(self.sigma_inner)
        dl_inner_B = params.dB * self.alpha_inner

        d_inner_A = jnp.flip(1 - (self.alpha_inner + self.sigma_inner))
        d_inner_B = 1 - params.dB * (self.alpha_inner + self.sigma_inner)

        du_inner_A = jnp.flip(self.alpha_inner)
        du_inner_B = params.dB * self.sigma_inner

        def stepper(c_prev: Scalar, x: ScanInputSequence) -> Tuple[Scalar, Scalar]:
            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    dl_inner_A,
                    jnp.array([-1.0, x.alpha_B0]),
                    dl_inner_B,
                    jnp.array([0.0]),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([1.0]),
                    d_inner_A,
                    jnp.array([x.beta_A0, x.beta_B0]),
                    d_inner_B,
                    jnp.array([1.0]),
                ]
            )

            du = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    du_inner_A,
                    jnp.array([x.alpha_A0, -1.0]),
                    du_inner_B,
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

    def solve(self, params: ElectronReactionParams) -> Scalar:
        stepper = self.create_stepper(params)

        c_init = jnp.concat([jnp.ones_like(self.X), jnp.zeros_like(self.X)])

        K_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.Ef)
        )

        K_ox = params.K0 * jnp.exp(
            (1 - params.alpha) * (self.applied_potentials - params.Ef)
        )

        alpha_A0 = -self.h0 * K_ox
        alpha_B0 = -self.h0 * K_red / params.dB

        beta_A0 = 1 + self.h0 * K_red
        beta_B0 = 1 + self.h0 * K_ox / params.dB

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
