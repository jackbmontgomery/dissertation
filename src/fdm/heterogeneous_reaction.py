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
    K1_red: Scalar
    K1_ox: Scalar
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
        dtheta: float = 1e-1,
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

    def compute_current(
        self,
        c: Array,
        K1_red: Scalar,
        K1_ox: Scalar,
        K2_red: Scalar,
        K2_ox: Scalar,
    ) -> Scalar:
        c0_A = c[2 * self.Nx - 2]
        c0_B = c[2 * self.Nx - 1]
        c0_C = c[2 * self.Nx]
        c0_D = c[2 * self.Nx + 1]

        return -(K1_red * c0_A - K1_ox * c0_B + K2_red * c0_C - K2_ox * c0_D)

    def create_stepper(self, params: HeterogenousReactionParams):
        sigma_inner_AB = jnp.flip(self.sigma_inner)
        alpha_inner_AB = jnp.flip(self.alpha_inner)

        sigma_inner_CD = self.sigma_inner
        alpha_inner_CD = self.alpha_inner

        d2l = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(sigma_inner_AB, params.dB * sigma_inner_AB),
                jnp.array([-1.0, -1.0]),
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(
                    params.dC * alpha_inner_CD, params.dD * alpha_inner_CD
                ),
                jnp.array([0.0, 0.0]),
            ]
        )

        d2u = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(alpha_inner_AB, params.dB * alpha_inner_AB),
                jnp.array([0.0, 0.0]),
                jnp.array([-1.0, -1.0]),
                interleave_concat_2d(
                    params.dC * sigma_inner_CD, params.dD * sigma_inner_CD
                ),
                jnp.array([0.0, 0.0]),
            ]
        )

        n = self.Nx - 2
        dl_inner_AB = jnp.zeros((2 * n,))
        dl_inner_CD = jnp.zeros((2 * n,))

        du_inner_AB = jnp.zeros((2 * n,))
        du_inner_CD = jnp.zeros((2 * n,))

        d_inner_AB = interleave_concat_2d(
            1 - (alpha_inner_AB + sigma_inner_AB),
            1 - params.dB * (alpha_inner_AB + sigma_inner_AB),
        )

        d_inner_CD = interleave_concat_2d(
            1 - params.dC * (alpha_inner_CD + sigma_inner_CD),
            1 - params.dD * (alpha_inner_CD + sigma_inner_CD),
        )

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

            current = self.compute_current(c, x.K1_red, x.K1_ox, x.K2_red, x.K2_ox)

            return c, current

        return stepper

    def solve(self, params: HeterogenousReactionParams) -> Scalar:
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

        B0_A_coef = -self.h0 * K1_red / params.dB
        C0_B_coef = jnp.full_like(B0_A_coef, -self.h0 * params.K_het / params.dC)
        D0_C_coef = -self.h0 * K2_red / params.dD

        beta_A_coef = 1.0 + self.h0 * K1_red
        beta_B_coef = 1.0 + self.h0 * (K1_ox + params.K_het) / params.dB
        beta_C_coef = 1.0 + self.h0 * K2_red / params.dC
        beta_D_coef = 1.0 + self.h0 * K2_ox / params.dD

        A0_B_coef = -self.h0 * K1_ox
        C0_D_coef = -self.h0 * K2_ox / params.dC

        xs = ScanInputSequence(
            K1_red=K1_red,
            K1_ox=K1_ox,
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

        _, current = scan(stepper, c_init, xs)

        return current
