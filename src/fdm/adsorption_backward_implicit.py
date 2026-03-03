from typing import Callable

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.fdm.base import AbstractFDSolver, setup_fd_discritisation
from src.params import AdsorptionReactionParams
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique


@dataclass
class ScanInputSequence:
    K_red_ads: Scalar
    K_ox_ads: Scalar
    K_red_sol: Scalar
    K_ox_sol: Scalar


class AdsorptionReactionBackwardImplicitFDSolver(AbstractFDSolver):
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

    def create_stepper(self, params: AdsorptionReactionParams) -> Callable:
        def stepper(sol_prev: Scalar, x: ScanInputSequence):
            CA_phiA_coef = (
                self.h0 * params.K_A_des + self.h0 * params.K_A_ads * sol_prev[2]
            )
            CA_phiB_coef = self.h0 * params.K_A_ads * sol_prev[2]
            CA_CA_coef = (
                -1.0
                - self.h0 * x.K_red_sol
                - self.h0 * params.K_A_ads * (1.0 - sol_prev[0] - sol_prev[1])
            )
            CA_CB_coef = self.h0 * x.K_ox_sol
            CA_CA1_coef = 1.0

            CB_phiA_coef = self.h0 * params.K_B_ads * sol_prev[3] / params.dB
            CB_phiB_coef = (
                self.h0 * params.K_B_des / params.dB
                + self.h0 * params.K_B_ads * sol_prev[3] / params.dB
            )
            CB_CA_coef = self.h0 * x.K_red_sol / params.dB
            CB_CB_coef = (
                -1.0
                - self.h0 * x.K_ox_sol / params.dB
                - self.h0
                * params.K_B_ads
                * (1.0 - sol_prev[0] - sol_prev[1])
                / params.dB
            )
            CB_CB1_coef = 1.0

            CA_multiplier = CB_phiA_coef / CA_phiA_coef

            d2l = jnp.concat(
                [
                    jnp.array(
                        [
                            0.0,
                            0.0,
                            CA_phiA_coef,
                            CB_phiB_coef - CA_multiplier * CA_phiB_coef,
                        ]
                    ),
                    interleave_concat_2d(
                        self.alpha_inner, params.dB * self.alpha_inner
                    ),
                    jnp.array([0.0, 0.0]),
                ]
            )

            dl = jnp.concat(
                [
                    jnp.array(
                        [
                            0.0,
                            -self.dt * x.K_red_ads
                            + self.dt * params.K_B_ads * self.beta * sol_prev[3],
                            CA_phiB_coef,
                            CB_CA_coef - CA_multiplier * CA_CA_coef,
                        ]
                    ),
                    jnp.zeros(2 * self.Nx - 2),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array(
                        [
                            1.0
                            + self.dt * x.K_red_ads
                            + self.dt * params.K_A_des * self.beta
                            + self.dt * params.K_A_ads * self.beta * sol_prev[2],
                            1.0
                            + self.dt * x.K_ox_ads
                            + self.dt * params.K_B_des * self.beta
                            + self.dt * params.K_B_ads * self.beta * sol_prev[3],
                            CA_CA_coef,
                            CB_CB_coef - CA_multiplier * CA_CB_coef,
                        ]
                    ),
                    interleave_concat_2d(
                        1 - (self.alpha_inner + self.sigma_inner),
                        1 - params.dB * (self.alpha_inner + self.sigma_inner),
                    ),
                    jnp.array([1.0, 1.0]),
                ]
            )

            du = jnp.concat(
                [
                    jnp.array(
                        [
                            -self.dt * x.K_ox_ads
                            + self.dt * params.K_A_ads * self.beta * sol_prev[2],
                            0.0,
                            CA_CB_coef,
                            -CA_multiplier * CA_CA1_coef,
                        ]
                    ),
                    jnp.zeros(2 * self.Nx - 2),
                ]
            )

            d2u = jnp.concat(
                [
                    jnp.array(
                        [
                            -self.dt
                            * params.K_A_ads
                            * self.beta
                            * (1 - sol_prev[0] - sol_prev[1]),
                            -self.dt
                            * params.K_B_ads
                            * self.beta
                            * (1 - sol_prev[0] - sol_prev[1]),
                            CA_CA1_coef,
                            CB_CB1_coef,
                        ]
                    ),
                    interleave_concat_2d(
                        self.sigma_inner, params.dB * self.sigma_inner
                    ),
                    jnp.array([0.0, 0.0]),
                ]
            )

            phiA_rhs = sol_prev[0] + self.dt * params.K_A_ads * self.beta * sol_prev[
                2
            ] * (sol_prev[0] + sol_prev[1])

            phiB_rhs = sol_prev[1] + self.dt * params.K_B_ads * self.beta * sol_prev[
                3
            ] * (sol_prev[0] + sol_prev[1])

            CA_rhs = (
                self.h0 * params.K_A_ads * sol_prev[2] * (sol_prev[0] + sol_prev[1])
            )
            CB_rhs = (
                self.h0
                * params.K_B_ads
                * sol_prev[3]
                * (sol_prev[0] + sol_prev[1])
                / params.dB
            )

            rhs = jnp.concat(
                [
                    jnp.array(
                        [phiA_rhs, phiB_rhs, CA_rhs, CB_rhs - CA_multiplier * CA_rhs]
                    ),
                    sol_prev[4:-2],
                    jnp.array([1.0, 0.0]),
                ]
            )

            sol = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            current = self.compute_current(sol, x)
            return sol, (sol, current)

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

        _, (sol, current) = scan(stepper, init_sol, xs)

        return sol, current
