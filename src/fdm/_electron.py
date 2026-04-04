from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.linear_solvers import tridiagonal_solve
from src.params import ElectronReactionParams
from src.voltammetry import AbstractVoltammetryTechnique

from ._base import AbstractFDSolver, setup_fd_discritisation


@dataclass
class ScanInputSequence:
    delta0: Scalar
    beta0: Scalar


class ElectronReactionFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Array
    Nx: int
    h0: Scalar
    dl: Scalar
    du: Scalar
    d_right: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float = 1e-3,
        omega: float = 1.1,
        dtheta: float = 2e-1,
    ):
        T, dt, X, alpha_inner, gamma_inner = setup_fd_discritisation(
            voltammetry, dtheta, h0, omega
        )

        self.X = X
        self.Nx = len(X)
        self.dt = dt
        self.applied_potentials = vmap(voltammetry.applied_potential)(T)
        self.h0 = jnp.array(h0)

        self.dl = jnp.concat([alpha_inner, jnp.array([0.0])])
        self.du = jnp.concatenate([jnp.array([-1.0]), gamma_inner])

        self.d_right = jnp.concat([1 - (alpha_inner + gamma_inner), jnp.ones(1)])

    def compute_current(self, c: Array) -> Scalar:
        c0_A = c[:, 0]
        c1_A = c[:, 1]
        c2_A = c[:, 2]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcA_dx = (h2**2 * (c0_A - c1_A) + h1**2 * (c2_A - c0_A)) / (h1 * h2 * (h1 - h2))

        return -dcA_dx

    def create_stepper(
        self,
        params: ElectronReactionParams,
    ) -> Callable[
        [Scalar, ScanInputSequence],
        Tuple[Scalar, Scalar],
    ]:
        def stepper(c_prev: Scalar, x: ScanInputSequence) -> Tuple[Scalar, Scalar]:
            d = jnp.concat([jnp.array([x.beta0]), self.d_right])

            rhs = jnp.concat(
                [
                    jnp.array([x.delta0]),
                    c_prev[1:-1],
                    jnp.array([1.0]),
                ]
            )

            c = tridiagonal_solve(self.dl, d, self.du, rhs)

            return c, c

        return stepper

    def solve(self, params: ElectronReactionParams) -> Tuple[Scalar, Scalar]:
        stepper = self.create_stepper(params)

        c_init = jnp.ones_like(self.X)

        K_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.thetaf)
        )

        K_ox = params.K0 * jnp.exp(
            (1 - params.alpha) * (self.applied_potentials - params.thetaf)
        )

        beta0 = 1 + self.h0 * (K_red + K_ox)
        delta0 = self.h0 * K_ox

        xs = ScanInputSequence(
            delta0=delta0,
            beta0=beta0,
        )

        _, solution = scan(stepper, c_init, xs)

        current = self.compute_current(solution)

        return solution, current
