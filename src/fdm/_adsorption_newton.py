from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import jit, vmap
from jax.lax import scan, while_loop
from jaxtyping import Array, Scalar

from src.linear_solvers import pentadiagonal_solve
from src.params import AdsorptionReactionParams
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from ._base import AbstractFDSolver, setup_fd_discritisation


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
    X: Scalar
    Nx: int
    alpha_inner: Scalar
    sigma_inner: Scalar
    dt: float
    h0: Scalar
    atol: float
    rtol: float
    _d2l_template: Scalar
    _dl_template: Scalar
    _d_template: Scalar
    _du_template: Scalar
    _d2u_template: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float = 1e-6,
        omega: float = 1.1,
        dtheta: float = 4e-3,
        atol: float = 1e-8,
        rtol: float = 1e-6,
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

        self.atol = atol
        self.rtol = rtol

        # Dimensionless Saturation Parameter
        self.beta = 1.0

        N = 2 + 2 * self.Nx
        n_inner = 2 * (self.Nx - 2)

        alpha_interleaved = interleave_concat_2d(alpha_inner, alpha_inner)
        sigma_interleaved = interleave_concat_2d(sigma_inner, sigma_inner)
        d_inner = interleave_concat_2d(
            1 - (alpha_inner + sigma_inner),
            1 - (alpha_inner + sigma_inner),
        )

        self._d2l_template = jnp.zeros(N - 2)
        self._d2l_template = self._d2l_template.at[2 : 2 + n_inner].set(
            alpha_interleaved
        )

        self._dl_template = jnp.zeros(N - 1)

        self._d_template = jnp.zeros(N)
        self._d_template = self._d_template.at[4 : 4 + n_inner].set(d_inner)
        self._d_template = self._d_template.at[-2].set(1.0)
        self._d_template = self._d_template.at[-1].set(1.0)

        self._du_template = jnp.zeros(N - 1)

        self._d2u_template = jnp.zeros(N - 2)
        self._d2u_template = self._d2u_template.at[2].set(-1.0)
        self._d2u_template = self._d2u_template.at[3].set(-1.0)
        self._d2u_template = self._d2u_template.at[4 : 4 + n_inner].set(
            sigma_interleaved
        )

    def compute_current(self, sol: Array) -> Scalar:
        c0_A = sol[:, 1]
        c1_A = sol[:, 2]
        c2_A = sol[:, 3]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcA_dx = (h2**2 * (c0_A - c1_A) + h1**2 * (c2_A - c0_A)) / (h1 * h2 * (h1 - h2))
        dgA_dt = jnp.gradient(sol[:, 0], self.dt)

        return -(dcA_dx - dgA_dt / self.beta)  # ty: ignore[unsupported-operator]

    def create_build_J_diags(
        self, params: AdsorptionReactionParams, x: ScanInputSequence
    ) -> Callable[[Scalar], Tuple[Scalar, Scalar, Scalar, Scalar, Scalar]]:
        h0_K_A_ads = self.h0 * params.K_A_ads
        h0_K_B_ads = self.h0 * params.K_B_ads
        h0_K_A_des = self.h0 * params.K_A_des
        h0_K_B_des = self.h0 * params.K_B_des
        dt_K_A_ads_beta = self.dt * params.K_A_ads * self.beta
        dt_K_B_ads_beta = self.dt * params.K_B_ads * self.beta
        dt_K_A_des_beta = self.dt * params.K_A_des * self.beta
        dt_K_B_des_beta = self.dt * params.K_B_des * self.beta
        h0_K_red_sol = self.h0 * x.K_red_sol
        h0_K_ox_sol = self.h0 * x.K_ox_sol
        dt_K_red_ads = self.dt * x.K_red_ads
        dt_K_ox_ads = self.dt * x.K_ox_ads

        def build_J_diags(sol: Scalar) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar]:
            phi_sum = sol[0] + sol[1]
            coverage_rem = 1.0 - phi_sum

            df0_dphiA = -dt_K_red_ads - dt_K_A_ads_beta * sol[2] - dt_K_A_des_beta - 1.0
            df0_dphiB = dt_K_ox_ads - dt_K_A_ads_beta * sol[2]
            df0_dCA = dt_K_A_ads_beta * coverage_rem

            df1_dphiA = dt_K_red_ads - dt_K_B_ads_beta * sol[3]
            df1_dphiB = -dt_K_ox_ads - dt_K_B_ads_beta * sol[3] - dt_K_B_des_beta - 1.0
            df1_dCB = dt_K_B_ads_beta * coverage_rem

            df2_dphiA = -h0_K_A_ads * sol[2] - h0_K_A_des
            df2_dphiB = -h0_K_A_ads * sol[2]
            df2_dCA = 1.0 + h0_K_red_sol + h0_K_A_ads * coverage_rem
            df2_dCB = -h0_K_ox_sol
            df2_dCA1 = -1.0

            df3_dphiA = -h0_K_B_ads * sol[3]
            df3_dphiB = -h0_K_B_ads * sol[3] - h0_K_B_des
            df3_dCA = -h0_K_red_sol
            df3_dCB = 1.0 + h0_K_ox_sol + h0_K_B_ads * coverage_rem
            df3_dCB1 = -1.0

            f2_multiplier = df3_dphiA / df2_dphiA

            d2l = self._d2l_template.at[0].set(df2_dphiA)
            d2l = d2l.at[1].set(df3_dphiB - f2_multiplier * df2_dphiB)

            dl = self._dl_template.at[0].set(df1_dphiA)
            dl = dl.at[1].set(df2_dphiB)
            dl = dl.at[2].set(df3_dCA - f2_multiplier * df2_dCA)

            d = self._d_template.at[0].set(df0_dphiA)
            d = d.at[1].set(df1_dphiB)
            d = d.at[2].set(df2_dCA)
            d = d.at[3].set(df3_dCB - f2_multiplier * df2_dCB)

            du = self._du_template.at[0].set(df0_dphiB)
            du = du.at[2].set(df2_dCB)
            du = du.at[3].set(-f2_multiplier * df2_dCA1)

            d2u = self._d2u_template.at[0].set(df0_dCA)
            d2u = d2u.at[1].set(df1_dCB)

            return d2l, dl, d, du, d2u

        return build_J_diags

    def create_build_F(
        self, params: AdsorptionReactionParams, sol_prev: Scalar, x: ScanInputSequence
    ) -> Callable[[Scalar], Scalar]:
        h0_K_A_ads = self.h0 * params.K_A_ads
        h0_K_B_ads = self.h0 * params.K_B_ads
        h0_K_A_des = self.h0 * params.K_A_des
        h0_K_B_des = self.h0 * params.K_B_des
        dt_K_A_ads_beta = self.dt * params.K_A_ads * self.beta
        dt_K_B_ads_beta = self.dt * params.K_B_ads * self.beta
        dt_K_A_des_beta = self.dt * params.K_A_des * self.beta
        dt_K_B_des_beta = self.dt * params.K_B_des * self.beta
        h0_K_red_sol = self.h0 * x.K_red_sol
        h0_K_ox_sol = self.h0 * x.K_ox_sol
        dt_K_red_ads = self.dt * x.K_red_ads
        dt_K_ox_ads = self.dt * x.K_ox_ads

        def build_F(sol: Scalar) -> Scalar:
            phi_sum = sol[0] + sol[1]
            coverage_rem = 1.0 - phi_sum

            f0 = (
                sol_prev[0]
                - dt_K_red_ads * sol[0]
                + dt_K_ox_ads * sol[1]
                + dt_K_A_ads_beta * sol[2] * coverage_rem
                - dt_K_A_des_beta * sol[0]
                - sol[0]
            )

            f1 = (
                sol_prev[1]
                + dt_K_red_ads * sol[0]
                - dt_K_ox_ads * sol[1]
                + dt_K_B_ads_beta * sol[3] * coverage_rem
                - dt_K_B_des_beta * sol[1]
                - sol[1]
            )

            f2 = (
                h0_K_red_sol * sol[2]
                - h0_K_ox_sol * sol[3]
                + h0_K_A_ads * sol[2] * coverage_rem
                - h0_K_A_des * sol[0]
                - sol[4]
                + sol[2]
            )

            f3 = (
                sol[3]
                - h0_K_red_sol * sol[2]
                + h0_K_ox_sol * sol[3]
                + h0_K_B_ads * sol[3] * coverage_rem
                - h0_K_B_des * sol[1]
                - sol[5]
            )

            f_CA_inner = (
                self.alpha_inner * sol[2:-4:2]
                + (1 - (self.alpha_inner + self.sigma_inner)) * sol[4:-2:2]
                + self.sigma_inner * sol[6::2]
                - sol_prev[4:-2:2]
            )

            f_CB_inner = (
                self.alpha_inner * sol[3:-4:2]
                + (1 - (self.alpha_inner + self.sigma_inner)) * sol[5:-2:2]
                + self.sigma_inner * sol[7::2]
                - sol_prev[5:-2:2]
            )

            f_CA_final = sol[-2] - 1.0
            f_CB_final = sol[-1]

            df2_dphiA = -h0_K_A_ads * sol[2] - h0_K_A_des
            df3_dphiA = -h0_K_B_ads * sol[3]
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
        build_F = self.create_build_F(params, sol_prev, x)
        build_J_diags = self.create_build_J_diags(params, x)

        def cond_fun(y: WhileOpArgs):
            delta_inf = jnp.max(jnp.abs(y.delta_sol))
            sol_inf = jnp.max(jnp.abs(y.sol))
            tol = self.atol + self.rtol * sol_inf
            return delta_inf > tol

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
            return sol, jnp.array([sol[0], sol[2], sol[4], sol[6]])

        return stepper

    @jit(static_argnums=(0,))
    def solve(self, params: AdsorptionReactionParams) -> Scalar:
        stepper = self.create_stepper(params)

        K_A_eq = params.K_A_ads / params.K_A_des

        phiA_init = K_A_eq / (1.0 + K_A_eq)
        phiB_init = 0.0

        phi_init = jnp.array([phiA_init, phiB_init])

        c_init = interleave_concat_2d(jnp.ones(self.Nx), jnp.zeros(self.Nx))

        init_sol = jnp.concat([phi_init, c_init])

        K_red_ads = params.K0_ads * jnp.exp(
            -params.alpha_ads * (self.applied_potentials - params.thetaf_ads)
        )

        K_ox_ads = params.K0_ads * jnp.exp(
            (1 - params.alpha_ads) * (self.applied_potentials - params.thetaf_ads)
        )

        K_red_sol = params.K0_sol * jnp.exp(
            -params.alpha_sol * (self.applied_potentials - params.thetaf_sol)
        )

        K_ox_sol = params.K0_sol * jnp.exp(
            (1 - params.alpha_sol) * (self.applied_potentials - params.thetaf_sol)
        )

        xs = ScanInputSequence(
            K_red_ads=K_red_ads,
            K_ox_ads=K_ox_ads,
            K_red_sol=K_red_sol,
            K_ox_sol=K_ox_sol,
        )

        _, c_surface_sol = scan(stepper, init_sol, xs)

        current = self.compute_current(c_surface_sol)

        return current
