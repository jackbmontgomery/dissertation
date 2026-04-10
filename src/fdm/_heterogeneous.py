from functools import partial

import jax.numpy as jnp
from chex import dataclass
from jax import jit, vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.linear_solvers import pentadiagonal_solve
from src.params import HeterogenousReactionParams
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from ._base import AbstractFDSolver, setup_fd_discritisation


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
    dt: float
    h0: Scalar
    d2l: Scalar
    d2u: Scalar
    _dl_template: Scalar
    _d_template: Scalar
    _du_template: Scalar

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

        gamma_inner_AB = jnp.flip(sigma_inner)
        alpha_inner_AB = jnp.flip(alpha_inner)

        gamma_inner_CD = sigma_inner
        alpha_inner_CD = alpha_inner

        # Second Sub-Diagonal
        self.d2l = jnp.concat(
            [
                interleave_concat_2d(gamma_inner_AB, gamma_inner_AB),
                jnp.array([-1.0, -1.0, 0.0, 0.0]),
                interleave_concat_2d(alpha_inner_CD, alpha_inner_CD),
                jnp.array([0.0, 0.0]),
            ]
        )

        # Second Super-Diagonal
        self.d2u = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(alpha_inner_AB, alpha_inner_AB),
                jnp.array([0.0, 0.0, -1.0, -1.0]),
                interleave_concat_2d(gamma_inner_CD, gamma_inner_CD),
            ]
        )

        # Sub-diagonal template: all zeros except 3 boundary entries set in stepper
        self._dl_template = jnp.zeros(4 * self.Nx - 1)

        # Super-diagonal template: all zeros except 2 boundary entries set in stepper
        self._du_template = jnp.zeros(4 * self.Nx - 1)

        # Diagonal template: inner values fixed, 4 boundary entries set in stepper
        d_AB_left = jnp.concat(
            [
                jnp.ones(2),
                interleave_concat_2d(
                    1 - (alpha_inner_AB + gamma_inner_AB),
                    1 - (alpha_inner_AB + gamma_inner_AB),
                ),
            ]
        )
        d_CD_right = jnp.concat(
            [
                interleave_concat_2d(
                    1 - (alpha_inner_CD + gamma_inner_CD),
                    1 - (alpha_inner_CD + gamma_inner_CD),
                ),
                jnp.ones(2),
            ]
        )
        self._d_template = jnp.concat([d_AB_left, jnp.zeros(4), d_CD_right])

    def compute_current(
        self, c_surf: Array, params: HeterogenousReactionParams
    ) -> Scalar:
        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]
        denom = h1 * h2 * (h1 - h2)

        dcA_dx = (
            h2**2 * (c_surf[:, 0] - c_surf[:, 1])
            + h1**2 * (c_surf[:, 2] - c_surf[:, 0])
        ) / denom
        dcC_dx = (
            h2**2 * (c_surf[:, 3] - c_surf[:, 4])
            + h1**2 * (c_surf[:, 5] - c_surf[:, 3])
        ) / denom

        K_het_cB = params.K_het * c_surf[:, 6]

        return -(dcA_dx + dcC_dx + K_het_cB)

    def create_stepper(self, params: HeterogenousReactionParams):
        N = self.Nx
        surface_indices = jnp.array(
            [
                2 * N - 2,  # A0  (c_surf[:, 0])
                2 * N - 4,  # A1  (c_surf[:, 1])
                2 * N - 6,  # A2  (c_surf[:, 2])
                2 * N,  # C0  (c_surf[:, 3])
                2 * N + 2,  # C1  (c_surf[:, 4])
                2 * N + 4,  # C2  (c_surf[:, 5])
                2 * N - 1,  # B0  (c_surf[:, 6])
            ]
        )

        def stepper(c_prev: Array, x: ScanInputSequence):
            dl = self._dl_template.at[2 * N - 2].set(x.B0_A_coef)
            dl = dl.at[2 * N - 1].set(x.C0_B_coef)
            dl = dl.at[2 * N].set(x.D0_C_coef)

            d = self._d_template.at[2 * N - 2].set(x.beta_A_coef)
            d = d.at[2 * N - 1].set(x.beta_B_coef)
            d = d.at[2 * N].set(x.beta_C_coef)
            d = d.at[2 * N + 1].set(x.beta_D_coef)

            du = self._du_template.at[2 * N - 2].set(x.A0_B_coef)
            du = du.at[2 * N].set(x.C0_D_coef)

            rhs = c_prev.at[0].set(1.0)
            rhs = rhs.at[1].set(0.0)
            rhs = rhs.at[2 * N - 2 : 2 * N + 2].set(jnp.zeros(4))
            rhs = rhs.at[-2:].set(jnp.zeros(2))

            c = pentadiagonal_solve(self.d2l, dl, d, du, self.d2u, rhs)

            return c, c[surface_indices]

        return stepper

    @partial(jit, static_argnums=(0,))
    def solve(self, params: HeterogenousReactionParams) -> Scalar:
        stepper = self.create_stepper(params)

        c_init = jnp.concat(
            [
                interleave_concat_2d(jnp.ones_like(self.X), jnp.zeros_like(self.X)),
                interleave_concat_2d(jnp.zeros_like(self.X), jnp.zeros_like(self.X)),
            ]
        )

        K1_red = params.K0_1 * jnp.exp(
            -params.alpha_1 * (self.applied_potentials - params.thetaf_1)
        )

        K1_ox = params.K0_1 * jnp.exp(
            (1.0 - params.alpha_1) * (self.applied_potentials - params.thetaf_1)
        )

        K2_red = params.K0_2 * jnp.exp(
            -params.alpha_2 * (self.applied_potentials - params.thetaf_2)
        )

        K2_ox = params.K0_2 * jnp.exp(
            (1.0 - params.alpha_2) * (self.applied_potentials - params.thetaf_2)
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

        current = self.compute_current(solution, params)

        return current
