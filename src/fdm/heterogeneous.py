from typing import Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import HeterogenousReactionParams
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver, setup_fd_discritisation


@dataclass
class ScanInputSequence:
    K2_red: Scalar
    K2_ox: Scalar
    B0_A_coef: Scalar
    C0_B_coef: Scalar
    D0_C_coef: Scalar
    beta_A_coef: Scalar
    beta_B_coef: Scalar
    beta_C_coef: Scalar
    beta_D_coef: Scalar
    A0_B_coef: Scalar
    C0_D_coef: Scalar


class HeterogeneousReactionFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Scalar
    Nx: int
    alpha_inner: Scalar
    sigma_inner: Scalar
    dt: float
    h0: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float = 1e-3,
        omega: float = 1.1,
        dtheta: float = 2e-1,
    ):
        T, dt, X, alpha_inner, sigma_inner = setup_fd_discritisation(
            voltammetry, dtheta, h0, omega
        )

        self.X = X
        self.Nx = len(X)
        self.dt = dt
        self.applied_potentials = vmap(voltammetry.applied_potential)(T)
        self.h0 = jnp.array(h0)
        self.alpha_inner = alpha_inner
        self.sigma_inner = sigma_inner

    def compute_current(self, c: Array, params: HeterogenousReactionParams) -> Scalar:
        c0_A = c[2 * self.Nx - 2]
        c1_A = c[2 * self.Nx - 4]
        c2_A = c[2 * self.Nx - 6]

        c0_C = c[2 * self.Nx]
        c1_C = c[2 * self.Nx + 2]
        c2_C = c[2 * self.Nx + 4]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcA_dx = (h2**2 * (c0_A - c1_A) + h1**2 * (c2_A - c0_A)) / (h1 * h2 * (h1 - h2))
        dcC_dx = (h2**2 * (c0_C - c1_C) + h1**2 * (c2_C - c0_C)) / (h1 * h2 * (h1 - h2))

        K_het_cB = params.K_het * c[2 * self.Nx - 1]

        return -(dcA_dx + dcC_dx + K_het_cB)

    def create_stepper(self, params: HeterogenousReactionParams):
        gamma_inner_AB = jnp.flip(self.sigma_inner)
        alpha_inner_AB = jnp.flip(self.alpha_inner)

        gamma_inner_CD = self.sigma_inner
        alpha_inner_CD = self.alpha_inner

        d_inner_AB = interleave_concat_2d(
            1 - (alpha_inner_AB + gamma_inner_AB),
            1 - (alpha_inner_AB + gamma_inner_AB),
        )

        d_inner_CD = interleave_concat_2d(
            1 - (alpha_inner_CD + gamma_inner_CD),
            1 - (alpha_inner_CD + gamma_inner_CD),
        )

        d2l = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(gamma_inner_AB, gamma_inner_AB),
                jnp.array([-1.0, -1.0]),
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(alpha_inner_CD, alpha_inner_CD),
                jnp.array([0.0, 0.0]),
            ]
        )

        d2u = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(alpha_inner_AB, alpha_inner_AB),
                jnp.array([0.0, 0.0]),
                jnp.array([-1.0, -1.0]),
                interleave_concat_2d(gamma_inner_CD, gamma_inner_CD),
                jnp.array([0.0, 0.0]),
            ]
        )

        n = self.Nx - 2
        dl_inner_AB = jnp.zeros(2 * n)
        dl_inner_CD = jnp.zeros(2 * n)

        du_inner_AB = jnp.zeros(2 * n)
        du_inner_CD = jnp.zeros(2 * n)

        def stepper(c_prev: Array, x: ScanInputSequence):
            dl = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),
                    dl_inner_AB,
                    jnp.array([0.0, x.B0_A_coef, x.C0_B_coef, x.D0_C_coef]),
                    dl_inner_CD,
                    jnp.array([0.0, 0.0]),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([1.0, 1.0]),
                    d_inner_AB,
                    jnp.array(
                        [x.beta_A_coef, x.beta_B_coef, x.beta_C_coef, x.beta_D_coef]
                    ),
                    d_inner_CD,
                    jnp.array([1.0, 1.0]),
                ]
            )

            du = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),
                    du_inner_AB,
                    jnp.array(
                        [
                            x.A0_B_coef,
                            0.0,
                            x.C0_D_coef,
                            0.0,
                        ]
                    ),
                    du_inner_CD,
                    jnp.array([0.0, 0.0]),
                ]
            )

            rhs = jnp.concat(
                [
                    jnp.array([1.0, 0.0]),
                    c_prev[2 : 2 * self.Nx - 2],
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                    c_prev[2 * self.Nx + 2 : -2],
                    jnp.array([0.0, 0.0]),
                ]
            )

            c = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            return c, c

        return stepper

    def solve(self, params: HeterogenousReactionParams) -> Tuple[Array, Scalar]:
        stepper = self.create_stepper(params)

        c_init = jnp.concat(
            [
                interleave_concat_2d(jnp.ones_like(self.X), jnp.zeros_like(self.X)),
                interleave_concat_2d(
                    jnp.full_like(self.X, 0.0), jnp.zeros_like(self.X)
                ),
            ]
        )

        K1_red = params.K0_1 * jnp.exp(
            -params.alpha_1 * (self.applied_potentials - params.Ef_1)
        )

        K1_ox = params.K0_1 * jnp.exp(
            (1.0 - params.alpha_1) * (self.applied_potentials - params.Ef_1)
        )

        K2_red = params.K0_2 * jnp.exp(
            -params.alpha_2 * (self.applied_potentials - params.Ef_2)
        )

        K2_ox = params.K0_2 * jnp.exp(
            (1.0 - params.alpha_2) * (self.applied_potentials - params.Ef_2)
        )

        B0_A_coef = -self.h0 * K1_red
        C0_B_coef = jnp.full_like(B0_A_coef, -self.h0 * params.K_het)
        D0_C_coef = -self.h0 * K2_red

        beta_A_coef = 1.0 + self.h0 * K1_red
        beta_B_coef = 1.0 + self.h0 * (K1_ox + params.K_het)
        beta_C_coef = 1.0 + self.h0 * K2_red
        beta_D_coef = 1.0 + self.h0 * K2_ox

        A0_B_coef = -self.h0 * K1_ox
        C0_D_coef = -self.h0 * K2_ox

        xs = ScanInputSequence(
            K2_red=K2_red,
            K2_ox=K2_ox,
            B0_A_coef=B0_A_coef,
            C0_B_coef=C0_B_coef,
            D0_C_coef=D0_C_coef,
            beta_A_coef=beta_A_coef,
            beta_B_coef=beta_B_coef,
            beta_C_coef=beta_C_coef,
            beta_D_coef=beta_D_coef,
            A0_B_coef=A0_B_coef,
            C0_D_coef=C0_D_coef,
        )

        _, solution = scan(stepper, c_init, xs)

        current = vmap(self.compute_current, in_axes=(0, None))(solution, params)

        return solution, current
