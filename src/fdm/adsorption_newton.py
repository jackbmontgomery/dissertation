from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan, while_loop
from jaxtyping import Array, Scalar

from src.fdm.base import AbstractFDSolver, setup_fd_discritisation
from src.params import AdsorptionReactionParams
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique


@dataclass
class WhileOpArgs:
    sol: Scalar
    delta_sol: Scalar


@dataclass
class ScanInputSequence:
    K_red_ads: Scalar
    K_ox_ads: Scalar
    K_red_sol: Scalar
    K_ox_sol: Scalar


class AdsorptionReactionNewtonFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
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

        # Dimensionless Saturation Parameter
        self.beta = 1.0

    def compute_current(self, sol: Array, x: ScanInputSequence) -> Scalar:
        gamma_A = sol[0]
        gamma_B = sol[1]
        cA_0 = sol[2]
        cB_0 = sol[3]

        return -(
            x.K_red_sol * cA_0
            - x.K_ox_sol * cB_0
            + x.K_red_ads * gamma_A
            - x.K_ox_ads * gamma_B
        )

    def create_build_J_diags(
        self, params: AdsorptionReactionParams, x: ScanInputSequence
    ) -> Callable[[Scalar], Tuple[Scalar, Scalar, Scalar, Scalar, Scalar]]:
        d_A_inner = 1 - (self.alpha_inner + self.sigma_inner)
        d_B_inner = 1 - params.dB * (self.alpha_inner + self.sigma_inner)

        def build_J_diags(sol: Scalar) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar]:
            df0_dphiA = (
                -self.dt * x.K_red_ads
                - self.dt * params.K_A_ads * self.beta * sol[2]
                - self.dt * params.K_A_des * self.beta
                - 1.0
            )

            df0_dphiB = (
                self.dt * x.K_ox_ads - self.dt * params.K_A_ads * self.beta * sol[2]
            )

            df0_dCA = self.dt * params.K_A_ads * self.beta * (1 - (sol[0] + sol[1]))

            df1_dphiA = (
                self.dt * x.K_red_ads - self.dt * params.K_B_ads * self.beta * sol[3]
            )

            df1_dphiB = (
                -self.dt * x.K_ox_ads
                - self.dt * params.K_B_ads * self.beta * sol[3]
                - self.dt * params.K_B_des * self.beta
                - 1.0
            )

            df1_dCB = self.dt * params.K_B_ads * self.beta * (1 - (sol[0] + sol[1]))

            df2_dphiA = -self.h0 * params.K_A_ads * sol[2] - self.h0 * params.K_A_des

            df2_dphiB = -self.h0 * params.K_A_ads * sol[2]

            df2_dCA = (
                1.0
                + self.h0 * x.K_red_sol
                + self.h0 * params.K_A_ads * (1 - (sol[0] + sol[1]))
            )

            df2_dCB = -self.h0 * x.K_ox_sol

            df2_dCA1 = -1.0

            df3_dphiA = -self.h0 * params.K_B_ads * sol[3] / params.dB

            df3_dphiB = (
                -self.h0 * params.K_B_ads * sol[3] / params.dB
                - self.h0 * params.K_B_des / params.dB
            )

            df3_dCA = -self.h0 * x.K_red_sol / params.dB

            df3_dCB = (
                1.0
                + self.h0 * x.K_ox_sol / params.dB
                + self.h0 * params.K_B_ads * (1 - (sol[0] + sol[1])) / params.dB
            )

            df3_dCB1 = -1.0

            f2_multiplier = df3_dphiA / df2_dphiA

            d2l = jnp.concat(
                [
                    jnp.array(
                        [0.0, 0.0, df2_dphiA, df3_dphiB - f2_multiplier * df2_dphiB]
                    ),
                    interleave_concat_2d(
                        self.alpha_inner, params.dB * self.alpha_inner
                    ),
                    jnp.array([0.0, 0.0]),
                ],
            )

            dl = jnp.concat(
                [
                    jnp.array(
                        [0.0, df1_dphiA, df2_dphiB, df3_dCA - f2_multiplier * df2_dCA]
                    ),
                    jnp.zeros(2 * self.Nx - 2),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array(
                        [
                            df0_dphiA,
                            df1_dphiB,
                            df2_dCA,
                            df3_dCB - f2_multiplier * df2_dCB,
                        ]
                    ),
                    interleave_concat_2d(d_A_inner, d_B_inner),
                    jnp.array([1.0, 1.0]),
                ]
            )

            du = jnp.concat(
                [
                    jnp.array([df0_dphiB, 0.0, df2_dCB, -f2_multiplier * df2_dCA1]),
                    jnp.zeros(2 * self.Nx - 2),
                ]
            )

            d2u = jnp.concat(
                [
                    jnp.array([df0_dCA, df1_dCB, df2_dCA1, df3_dCB1]),
                    interleave_concat_2d(
                        self.sigma_inner, params.dB * self.sigma_inner
                    ),
                    jnp.array([0.0, 0.0]),
                ]
            )

            return d2l, dl, d, du, d2u

        return build_J_diags

    def create_build_F(
        self, params: AdsorptionReactionParams, sol_prev: Scalar, x: ScanInputSequence
    ) -> Callable[[Scalar], Scalar]:
        def build_F(sol: Scalar) -> Scalar:
            f0 = (
                sol_prev[0]
                - self.dt * x.K_red_ads * sol[0]
                + self.dt * x.K_ox_ads * sol[1]
                + self.dt * params.K_A_ads * self.beta * sol[2] * (1 - sol[0] - sol[1])
                - self.dt * params.K_A_des * self.beta * sol[0]
                - sol[0]
            )

            f1 = (
                sol_prev[1]
                + self.dt * x.K_red_ads * sol[0]
                - self.dt * x.K_ox_ads * sol[1]
                + self.dt * params.K_B_ads * self.beta * sol[3] * (1 - sol[0] - sol[1])
                - self.dt * params.K_B_des * self.beta * sol[1]
                - sol[1]
            )

            f2 = (
                self.h0 * x.K_red_sol * sol[2]
                - self.h0 * x.K_ox_sol * sol[3]
                + self.h0 * params.K_A_ads * sol[2] * (1 - sol[0] - sol[1])
                - self.h0 * params.K_A_des * sol[0]
                - sol[4]
                + sol[2]
            )

            f3 = (
                sol[3]
                - self.h0 * x.K_red_sol * sol[2] / params.dB
                + self.h0 * x.K_ox_sol * sol[3] / params.dB
                + self.h0 * params.K_B_ads * sol[3] * (1 - sol[0] - sol[1]) / params.dB
                - self.h0 * params.K_B_des * sol[1] / params.dB
                - sol[5]
            )

            f_CA_inner = (
                self.alpha_inner * sol[2:-4:2]
                + (1 - (self.alpha_inner + self.sigma_inner)) * sol[4:-2:2]
                + self.sigma_inner * sol[6::2]
                - sol_prev[4:-2:2]
            )

            f_CB_inner = (
                params.dB * self.alpha_inner * sol[3:-4:2]
                + (1 - params.dB * (self.alpha_inner + self.sigma_inner)) * sol[5:-2:2]
                + params.dB * self.sigma_inner * sol[7::2]
                - sol_prev[5:-2:2]
            )

            f_CA_final = sol[-2] - 1.0
            f_CB_final = sol[-1]

            df2_dphiA = -self.h0 * params.K_A_ads * sol[2] - self.h0 * params.K_A_des

            df3_dphiA = -self.h0 * params.K_B_ads * sol[3] / params.dB

            f2_multiplier = df3_dphiA / df2_dphiA

            F = jnp.concat(
                [
                    jnp.array([f0, f1, f2, f3 - f2_multiplier * f2]),
                    interleave_concat_2d(f_CA_inner, f_CB_inner),
                    jnp.array([f_CA_final, f_CB_final]),
                ]
            )
            return F

        return build_F

    def newton_solve(
        self, sol_prev: Scalar, params: AdsorptionReactionParams, x: ScanInputSequence
    ):
        a_tol = 1e-8

        build_F = self.create_build_F(params, sol_prev, x)
        build_J_diags = self.create_build_J_diags(params, x)

        def cond_fun(y: WhileOpArgs):
            return jnp.less_equal(a_tol, jnp.max(jnp.abs(y.delta_sol)))

        def body_fun(y: WhileOpArgs):
            F = build_F(y.sol)

            d2l, dl, d, du, d2u = build_J_diags(y.sol)

            delta_sol = pentadiagonal_solve(d2l, dl, d, du, d2u, -F)

            sol = y.sol + delta_sol

            new_y = WhileOpArgs(sol=sol, delta_sol=delta_sol)
            return new_y

        y = WhileOpArgs(
            sol=sol_prev,
            delta_sol=jnp.ones(2 + 2 * self.Nx),
        )

        y = while_loop(cond_fun, body_fun, y)

        return y.sol

    def create_stepper(self, params: AdsorptionReactionParams) -> Callable:
        def stepper(sol_prev: Scalar, x: ScanInputSequence):
            sol = self.newton_solve(sol_prev, params, x)
            current = self.compute_current(sol, x)
            return sol, current

        return stepper

    def solve(self, params: AdsorptionReactionParams) -> Scalar:
        stepper = self.create_stepper(params)

        K_A_eq = params.K_A_ads / params.K_A_des

        phiA_init = K_A_eq / (1.0 + K_A_eq)
        phiB_init = 0.0

        phi_init = jnp.array([phiA_init, phiB_init])

        c_init = interleave_concat_2d(jnp.ones(self.Nx), jnp.zeros(self.Nx))

        init_sol = jnp.concat([phi_init, c_init])

        K_red_ads = params.K0_ads * jnp.exp(
            -params.alpha_ads * (self.applied_potentials - params.Ef_ads)
        )

        K_ox_ads = params.K0_ads * jnp.exp(
            (1 - params.alpha_ads) * (self.applied_potentials - params.Ef_ads)
        )

        K_red_sol = params.K0_sol * jnp.exp(
            -params.alpha_sol * (self.applied_potentials - params.Ef_sol)
        )

        K_ox_sol = params.K0_sol * jnp.exp(
            (1 - params.alpha_sol) * (self.applied_potentials - params.Ef_sol)
        )

        xs = ScanInputSequence(
            K_red_ads=K_red_ads,
            K_ox_ads=K_ox_ads,
            K_red_sol=K_red_sol,
            K_ox_sol=K_ox_sol,
        )

        _, current = scan(stepper, init_sol, xs)

        return current
