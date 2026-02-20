import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import SecondOrderECirreMechanismFDMParams
from src.solvers import pentadiagonal_solve, tridiagonal_solve
from src.utils import interleave_concat_2d, interleave_concat_4d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver


class SecondOrderECirreFDMSolverExplicitApprox(AbstractFDMSolver):
    applied_potentials: Scalar
    dt: float
    X: Scalar
    num_x: int
    h: Scalar
    alpha_inner: Scalar
    sigma_inner: Scalar

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
        self.num_x = len(X)

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, ck: Array) -> Array:
        c0 = ck[0]
        c1 = ck[4]
        c2 = ck[8]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcdx = (h2**2 * (c0 - c1) + h1**2 * (c2 - c0)) / (h1 * h2 * (h1 - h2))

        return -dcdx

    def _create_stepper(self, params: SecondOrderECirreMechanismFDMParams):
        def stepper(c_prev: Array, applied_potential: Scalar):
            gamma = (
                params.Kplus * self.dt * c_prev[5:-4:4] * c_prev[6:-4:4]
                - params.Kminus * self.dt * c_prev[4:-4:4] * c_prev[7:-4:4]
            )

            # --- AB Solve ---

            k_red = params.K0 * jnp.exp(-params.alpha * (applied_potential - params.E0))
            k_ox = params.K0 * jnp.exp(
                (1.0 - params.alpha) * (applied_potential - params.E0)
            )

            d2l_inner = interleave_concat_2d(
                self.alpha_inner, params.dB * self.alpha_inner
            )

            d2l = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),  # compatibility
                    d2l_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            # Lower Diagonal: Coefficients for A
            dlB_inner = jnp.zeros((self.num_x - 2,))
            dlA_inner = jnp.zeros((self.num_x - 2,))

            dl_inner = interleave_concat_2d(dlB_inner, dlA_inner)

            dl = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    jnp.array([-self.h * k_red / params.dB]),
                    dl_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            # Main Diagonal: beta_{n,A} -> beta_{0,A}, beta_{0,B} -> beta_{n,B}

            dA_inner = 1 - (self.alpha_inner + self.sigma_inner)

            dB_inner = 1 - params.dB * (self.alpha_inner + self.sigma_inner)

            d_inner = interleave_concat_2d(dA_inner, dB_inner)

            d = jnp.concat(
                [
                    jnp.array([1.0 + self.h * k_red, 1.0 + self.h * k_ox / params.dB]),
                    d_inner,
                    jnp.array([1.0, 1.0]),
                ]
            )

            # Upper Diagonal: alpha_{n-1, A} -> alpha_{0, A}, sigma_{0, B} -> sigma_{n-2, B}

            duB_inner = jnp.zeros((self.num_x - 2,))
            duA_inner = jnp.zeros((self.num_x - 2,))
            du_inner = interleave_concat_2d(duB_inner, duA_inner)

            du = jnp.concat(
                [
                    jnp.array([-self.h * k_ox, 0.0]),
                    du_inner,
                    jnp.array([0.0]),
                    jnp.array([0.0]),  # compatibility
                ]
            )

            d2u_inner = interleave_concat_2d(
                self.sigma_inner,
                params.dB * self.sigma_inner,
            )

            d2u = jnp.concatenate(
                [jnp.array([-1.0, -1.0]), d2u_inner, jnp.array([0.0, 0.0])]
            )  # compatibility

            rhsA_inner = jnp.concat(
                [jnp.array([0.0]), c_prev[4:-4:4] + gamma, jnp.array([1.0])]
            )

            rhsB_inner = jnp.concat(
                [jnp.array([0.0]), c_prev[5:-4:4] - gamma, jnp.array([0.0])]
            )

            rhs = interleave_concat_2d(rhsA_inner, rhsB_inner)

            ck_AB = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            # --- Y Solve ---

            dl = jnp.concat(
                [jnp.array([0.0]), params.dY * self.alpha_inner, jnp.array([0.0])]
            )
            d = jnp.concat(
                [
                    jnp.array([-1.0]),
                    1 - params.dY * (self.alpha_inner + self.sigma_inner),
                    jnp.array([1.0]),
                ]
            )
            du = jnp.concat(
                [jnp.array([1.0]), params.dY * self.sigma_inner, jnp.array([0.0])]
            )

            rhs = jnp.concat(
                [jnp.array([0.0]), c_prev[6:-4:4] - gamma, jnp.array([1.0])]
            )

            ck_Y = tridiagonal_solve(dl, d, du, rhs)

            # --- Z Solve ---

            dl = jnp.concat(
                [jnp.array([0.0]), params.dZ * self.alpha_inner, jnp.array([0.0])]
            )
            d = jnp.concat(
                [
                    jnp.array([-1.0]),
                    1 - params.dZ * (self.alpha_inner + self.sigma_inner),
                    jnp.array([1.0]),
                ]
            )
            du = jnp.concat(
                [jnp.array([1.0]), params.dZ * self.sigma_inner, jnp.array([0.0])]
            )

            rhs = jnp.concat(
                [jnp.array([0.0]), c_prev[7:-4:4] + gamma, jnp.array([0.0])]
            )

            ck_Z = tridiagonal_solve(dl, d, du, rhs)

            # --- Combine ---

            ck = interleave_concat_4d(ck_AB[::2], ck_AB[1::2], ck_Y, ck_Z)

            current = self.compute_current(ck)

            return ck, current

        return stepper

    def solve(self, params: SecondOrderECirreMechanismFDMParams) -> Scalar:
        c_init = interleave_concat_4d(
            jnp.ones_like(self.X),
            jnp.zeros_like(self.X),
            jnp.zeros_like(self.X),
            jnp.ones_like(self.X),
        )
        stepper = self._create_stepper(params)

        _, current = scan(stepper, c_init, self.applied_potentials)

        return current
