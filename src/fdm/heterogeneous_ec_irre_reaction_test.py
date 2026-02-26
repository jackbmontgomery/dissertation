import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import (
    ECirreMechanismFDMParams,
    EMechanismFDMParams,
    HeterogenousECirreMechanismFDMParams,
)
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver


@dataclass
class ScanInputSequence:
    K1_red: Scalar
    K1_ox: Scalar


class HeterogeneousECirreTestFDSolver(AbstractFDSolver):
    applied_potentials: Scalar
    X: Scalar
    Nx: int
    alpha_inner: Scalar
    sigma_inner: Scalar
    dt: float
    h: Scalar

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h: float = 5e-3,
        dtheta: float = 5e-2,
    ):
        # Suggestion from Understanding Voltammetry 3.4.1
        dt = dtheta / voltammetry.sigma
        self.dt = dt

        T = jnp.linspace(
            voltammetry.t_min,
            voltammetry.t_max,
            int((voltammetry.t_max - voltammetry.t_min) / dt),
        )

        self.applied_potentials = vmap(voltammetry.applied_potential)(T)

        # Einstein on Brownian Motion
        x_max = 6.0 * jnp.sqrt(voltammetry.t_max)
        X = jnp.linspace(0.0, x_max, int(x_max / h))
        self.X = X
        self.Nx = len(X)

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(
        self,
        c: Array,
        K1_red: Scalar,
        K1_ox: Scalar,
    ) -> Scalar:
        c0_A = c[2 * self.Nx - 2]
        c0_B = c[2 * self.Nx - 1]
        return -(K1_red * c0_A - K1_ox * c0_B)

    def create_stepper(self, params: EMechanismFDMParams):
        d2l = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(self.sigma_inner, params.dB * self.sigma_inner),
                jnp.array([-1.0, -1.0]),
            ]
        )

        d2u = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                interleave_concat_2d(self.alpha_inner, params.dB * self.alpha_inner),
                jnp.array([0.0, 0.0]),
            ]
        )

        n = self.Nx - 2
        dl_inner_AB = jnp.zeros((2 * n,))

        du_inner_AB = jnp.zeros((2 * n,))

        d_inner_AB = interleave_concat_2d(
            1 - (self.alpha_inner + self.sigma_inner),
            1 - params.dB * (self.alpha_inner + self.sigma_inner),
        )

        def stepper(c_prev: Array, x: ScanInputSequence):
            dl = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),
                    dl_inner_AB,
                    jnp.array(
                        [
                            0.0,
                            -self.h * x.K1_red / params.dB,
                        ]
                    ),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([1.0, 1.0]),
                    d_inner_AB,
                    jnp.array(
                        [
                            1.0 + self.h * x.K1_red,
                            1.0 + self.h * x.K1_ox / params.dB,
                        ]
                    ),
                ]
            )

            du = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),
                    du_inner_AB,
                    jnp.array(
                        [
                            -self.h * x.K1_ox,
                            0.0,
                        ]
                    ),
                ]
            )

            rhs = jnp.concat(
                [
                    jnp.array([1.0, 0.0]),
                    interleave_concat_2d(
                        c_prev[2 : 2 * self.Nx - 2 : 2],
                        c_prev[3 : 2 * self.Nx - 2 : 2],
                    ),
                    jnp.array([0.0, 0.0]),
                ]
            )
            print(d2l.shape, dl.shape, d.shape, du.shape, d2u.shape, rhs.shape)

            c = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            current = self.compute_current(c, x.K1_red, x.K1_ox)

            return c, current

        return stepper

    def solve(self, params: EMechanismFDMParams) -> Scalar:
        stepper = self.create_stepper(params)

        c_init = interleave_concat_2d(jnp.ones_like(self.X), jnp.zeros_like(self.X))

        K1_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.E0)
        )

        K1_ox = params.K0 * jnp.exp(
            (1.0 - params.alpha) * (self.applied_potentials - params.E0)
        )

        xs = ScanInputSequence(
            K1_red=K1_red,
            K1_ox=K1_ox,
        )

        _, current = scan(stepper, c_init, xs)

        return current
