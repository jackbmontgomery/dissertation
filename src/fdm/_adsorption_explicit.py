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


class AdsorptionReactionExplicitFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    Nx: int
    alpha_inner: Scalar
    gamma_inner: Scalar
    dt: float
    h0: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float = 1e-5,
        omega: float = 1.1,
        dtheta: float = 5e-2,
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

        self.beta = 1.0

        self.d_right = jnp.concat(
            [
                interleave_concat_2d(
                    1 - (alpha_inner + gamma_inner),
                    1 - (alpha_inner + gamma_inner),
                ),
                jnp.ones(2),
            ]
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
        d2l_inner = interleave_concat_2d(self.alpha_inner, self.alpha_inner)
        d2u_inner = interleave_concat_2d(self.gamma_inner, self.gamma_inner)

        N = 2 * self.Nx + 2

        d2l = jnp.zeros(N - 2)
        d2l = d2l.at[0].set(self.h0 * params.K_A_des)
        d2l = d2l.at[1].set(self.h0 * params.K_B_des)
        d2l = d2l.at[2 : 2 + len(d2l_inner)].set(d2l_inner)

        d2u = jnp.zeros(N - 2)
        d2u = d2u.at[0].set(-self.dt * params.K_A_ads * self.beta)
        d2u = d2u.at[1].set(-self.dt * params.K_B_ads * self.beta)
        d2u = d2u.at[2].set(1.0)
        d2u = d2u.at[3].set(1.0)
        d2u = d2u.at[4 : 4 + len(d2u_inner)].set(d2u_inner)

        dl_template = jnp.zeros(N - 1)

        d_template = jnp.zeros(N)
        d_template = d_template.at[4:].set(self.d_right)

        du_template = jnp.zeros(N - 1)

        rhs_bc = jnp.zeros(N)
        rhs_bc = rhs_bc.at[-2].set(1.0)
        rhs_bc = rhs_bc.at[-1].set(0.0)

        dt_K_A_ads_beta = self.dt * params.K_A_ads * self.beta
        dt_K_B_ads_beta = self.dt * params.K_B_ads * self.beta
        h0_K_A_ads = self.h0 * params.K_A_ads
        h0_K_B_ads = self.h0 * params.K_B_ads
        dt_K_A_des_beta = self.dt * params.K_A_des * self.beta
        dt_K_B_des_beta = self.dt * params.K_B_des * self.beta

        def stepper(sol_prev: Scalar, x: ScanInputSequence):
            dl = dl_template.at[0].set(-self.dt * x.K_red_ads)
            dl = dl.at[2].set(self.h0 * x.K_red_sol)

            d = d_template.at[0].set(1.0 + self.dt * x.K_red_ads + dt_K_A_des_beta)
            d = d.at[1].set(1.0 + self.dt * x.K_ox_ads + dt_K_B_des_beta)
            d = d.at[2].set(-1.0 - self.h0 * x.K_red_sol - h0_K_A_ads)
            d = d.at[3].set(-1.0 - self.h0 * x.K_ox_sol - h0_K_B_ads)

            du = du_template.at[0].set(-self.dt * x.K_ox_ads)
            du = du.at[2].set(self.h0 * x.K_ox_sol)

            phi_sum = sol_prev[0] + sol_prev[1]

            rhs = rhs_bc.at[0].set(
                sol_prev[0] - dt_K_A_ads_beta * sol_prev[2] * phi_sum
            )
            rhs = rhs.at[1].set(sol_prev[1] - dt_K_B_ads_beta * sol_prev[3] * phi_sum)
            rhs = rhs.at[2].set(-h0_K_A_ads * sol_prev[2] * phi_sum)
            rhs = rhs.at[3].set(-h0_K_B_ads * sol_prev[3] * phi_sum)
            rhs = rhs.at[4:-2].set(sol_prev[4:-2])

            sol = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

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
