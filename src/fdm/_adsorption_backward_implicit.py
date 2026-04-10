from functools import partial
from typing import Callable

import jax.numpy as jnp
from chex import dataclass
from jax import jit, vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.linear_solvers import pentadiagonal_solve
from src.params import AdsorptionReactionParams
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from ._base import AbstractFDSolver, setup_fd_discritisation


@dataclass
class ScanInputSequence:
    K_red_ads: Scalar
    K_ox_ads: Scalar
    K_red_sol: Scalar
    K_ox_sol: Scalar


class AdsorptionReactionBackwardImplicitFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    Nx: int
    dt: float
    h0: Scalar
    _d2l_template: Scalar
    _dl_template: Scalar
    _d_template: Scalar
    _du_template: Scalar
    _d2u_template: Scalar

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
        self._d2u_template = self._d2u_template.at[2].set(1.0)
        self._d2u_template = self._d2u_template.at[3].set(1.0)
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

    def create_stepper(self, params: AdsorptionReactionParams) -> Callable:
        h0_K_A_des = self.h0 * params.K_A_des
        h0_K_B_des = self.h0 * params.K_B_des
        h0_K_A_ads = self.h0 * params.K_A_ads
        h0_K_B_ads = self.h0 * params.K_B_ads
        dt_K_A_des_beta = self.dt * params.K_A_des * self.beta
        dt_K_B_des_beta = self.dt * params.K_B_des * self.beta
        dt_K_A_ads_beta = self.dt * params.K_A_ads * self.beta
        dt_K_B_ads_beta = self.dt * params.K_B_ads * self.beta

        def stepper(sol_prev: Scalar, x: ScanInputSequence):
            phi_sum = sol_prev[0] + sol_prev[1]
            coverage_rem = 1.0 - phi_sum

            CA_phiA_coef = h0_K_A_des + h0_K_A_ads * sol_prev[2]
            CA_phiB_coef = h0_K_A_ads * sol_prev[2]
            CA_CA_coef = -1.0 - self.h0 * x.K_red_sol - h0_K_A_ads * coverage_rem
            CA_CB_coef = self.h0 * x.K_ox_sol
            CA_CA1_coef = 1.0

            CB_phiA_coef = h0_K_B_ads * sol_prev[3]
            CB_phiB_coef = h0_K_B_des + h0_K_B_ads * sol_prev[3]
            CB_CA_coef = self.h0 * x.K_red_sol
            CB_CB_coef = -1.0 - self.h0 * x.K_ox_sol - h0_K_B_ads * coverage_rem
            CB_CB1_coef = 1.0

            CA_multiplier = CB_phiA_coef / CA_phiA_coef

            d2l = self._d2l_template.at[0].set(CA_phiA_coef)
            d2l = d2l.at[1].set(CB_phiB_coef - CA_multiplier * CA_phiB_coef)

            dl = self._dl_template.at[0].set(
                -self.dt * x.K_red_ads + dt_K_B_ads_beta * sol_prev[3]
            )
            dl = dl.at[1].set(CA_phiB_coef)
            dl = dl.at[2].set(CB_CA_coef - CA_multiplier * CA_CA_coef)

            d = self._d_template.at[0].set(
                1.0
                + self.dt * x.K_red_ads
                + dt_K_A_des_beta
                + dt_K_A_ads_beta * sol_prev[2]
            )
            d = d.at[1].set(
                1.0
                + self.dt * x.K_ox_ads
                + dt_K_B_des_beta
                + dt_K_B_ads_beta * sol_prev[3]
            )
            d = d.at[2].set(CA_CA_coef)
            d = d.at[3].set(CB_CB_coef - CA_multiplier * CA_CB_coef)

            du = self._du_template.at[0].set(
                -self.dt * x.K_ox_ads + dt_K_A_ads_beta * sol_prev[2]
            )
            du = du.at[2].set(CA_CB_coef)
            du = du.at[3].set(-CA_multiplier * CA_CA1_coef)

            d2u = self._d2u_template.at[0].set(-dt_K_A_ads_beta * coverage_rem)
            d2u = d2u.at[1].set(-dt_K_B_ads_beta * coverage_rem)

            phiA_rhs = sol_prev[0] + dt_K_A_ads_beta * sol_prev[2] * phi_sum
            phiB_rhs = sol_prev[1] + dt_K_B_ads_beta * sol_prev[3] * phi_sum
            CA_rhs = h0_K_A_ads * sol_prev[2] * phi_sum
            CB_rhs = h0_K_B_ads * sol_prev[3] * phi_sum

            rhs = sol_prev.at[0].set(phiA_rhs)
            rhs = rhs.at[1].set(phiB_rhs)
            rhs = rhs.at[2].set(CA_rhs)
            rhs = rhs.at[3].set(CB_rhs - CA_multiplier * CA_rhs)
            rhs = rhs.at[-2].set(1.0)
            rhs = rhs.at[-1].set(0.0)

            sol = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            return sol, jnp.array([sol[0], sol[2], sol[4], sol[6]])

        return stepper

    @partial(jit, static_argnums=(0,))
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
