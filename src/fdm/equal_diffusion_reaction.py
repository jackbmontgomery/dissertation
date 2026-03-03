from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import EqualDiffusionReactionParams
from src.solvers import tridiagonal_solve
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver, setup_fd_discritisation


@dataclass
class ScanInputSequence:
    delta0: Scalar
    beta0: Scalar


class EqualDiffusionReactionFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Array
    Nx: int
    h0: Scalar
    hs: Scalar
    alpha_inner: Scalar
    gamma_inner: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float = 1e-3,
        omega: float = 1.1,
        dtheta: float = 1e-1,
    ):
        T, dt, X, alpha_inner, gamma_inner = setup_fd_discritisation(
            voltammetry, dtheta, h0, omega
        )

        self.X = X
        self.Nx = len(X)
        self.dt = dt
        self.applied_potentials = vmap(voltammetry.applied_potential)(T)
        self.h0 = jnp.array(h0)
        self.alpha_inner = alpha_inner
        self.gamma_inner = gamma_inner
        self.beta_inner = 1 - (self.alpha_inner + self.gamma_inner)

    def compute_current(self, c: Array) -> Scalar:
        c0_A = c[0]
        c1_A = c[1]
        c2_A = c[2]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcA_dx = (h2**2 * (c0_A - c1_A) + h1**2 * (c2_A - c0_A)) / (h1 * h2 * (h1 - h2))

        return -dcA_dx

    def create_stepper(
        self,
        params: EqualDiffusionReactionParams,
    ) -> Callable[
        [Scalar, ScanInputSequence],
        Tuple[Scalar, Scalar],
    ]:
        def stepper(c_prev: Scalar, x: ScanInputSequence) -> Tuple[Scalar, Scalar]:
            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    self.alpha_inner,
                    jnp.array([0.0]),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([x.beta0]),
                    self.beta_inner,
                    jnp.array([1.0]),
                ]
            )

            du = jnp.concatenate(
                [
                    jnp.array([-1.0]),
                    self.gamma_inner,
                    jnp.array([0.0]),  # compatibility
                ]
            )

            rhs = jnp.concat(
                [
                    jnp.array([x.delta0]),
                    c_prev[1:-1],
                    jnp.array([1.0]),
                ]
            )

            c = tridiagonal_solve(dl, d, du, rhs)

            return c, c

        return stepper

    def solve(self, params: EqualDiffusionReactionParams) -> Scalar:
        stepper = self.create_stepper(params)

        c_init = jnp.ones_like(self.X)

        K_red = params.K0 * jnp.exp(-params.alpha * self.applied_potentials)

        K_ox = params.K0 * jnp.exp((1 - params.alpha) * self.applied_potentials)

        beta0 = 1 + self.h0 * (K_red + K_ox)
        delta0 = self.h0 * K_ox

        xs = ScanInputSequence(
            delta0=delta0,
            beta0=beta0,
        )

        _, C = scan(stepper, c_init, xs)

        current = vmap(self.compute_current)(C)

        return current
