import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Scalar

from src.params import SecondOrderECirreMechanismFDMParams
from src.solvers import tridiagonal_solve
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver


@dataclass
class ScanInputSequence:
    k_red: Scalar
    k_ox: Scalar
    alpha_A0: Scalar
    alpha_B0: Scalar
    beta_A0: Scalar
    beta_B0: Scalar


@dataclass
class Concentration:
    A: Scalar
    B: Scalar
    Y: Scalar
    Z: Scalar


class SecondOrderECirreFDMSolverExplicit(AbstractFDSolver):
    applied_potentials: Scalar
    dt: float
    X: Scalar
    Nx: int
    h1: Scalar
    h2: Scalar
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
        self.Nx = len(X)

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h1 = X[1] - X[0]
        self.h2 = X[2] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, c: Concentration, k_red: Scalar, k_ox: Scalar) -> Scalar:
        return -(k_red * c.A[0] - k_ox * c.B[0])

    def _create_stepper(self, params: SecondOrderECirreMechanismFDMParams):
        dl_Y = jnp.concat(
            [
                jnp.array([0.0]),
                params.dY * self.alpha_inner,
                jnp.array([0]),
            ]
        )

        d_Y = jnp.concat(
            [
                jnp.array([-1.0]),
                1 - params.dY * (self.alpha_inner + self.sigma_inner),
                jnp.array([1.0]),
            ]
        )

        du_Y = jnp.concat(
            [
                jnp.array([1.0]),
                params.dY * self.sigma_inner,
                jnp.array([0.0]),
            ]
        )

        dl_Z = jnp.concat(
            [
                jnp.array([0.0]),
                params.dZ * self.alpha_inner,
                jnp.array([0]),
            ]
        )

        d_Z = jnp.concat(
            [
                jnp.array([-1.0]),
                1 - params.dZ * (self.alpha_inner + self.sigma_inner),
                jnp.array([1.0]),
            ]
        )

        du_Z = jnp.concat(
            [
                jnp.array([1.0]),
                params.dZ * self.sigma_inner,
                jnp.array([0.0]),
            ]
        )

        def stepper(c_prev: Concentration, x: ScanInputSequence):
            # --- Current AB Solve

            dl_AB = jnp.concat(
                [
                    jnp.array([0.0]),  # compatibility
                    self.sigma_inner,
                    jnp.array([-1.0, x.alpha_B0]),
                    params.dB * self.alpha_inner,
                    jnp.array([0.0]),
                ]
            )

            d_AB = jnp.concat(
                [
                    jnp.array([1.0]),
                    1 - (self.alpha_inner + self.sigma_inner),
                    jnp.array([x.beta_A0, x.beta_B0]),
                    1 - params.dB * (self.alpha_inner + self.sigma_inner),
                    jnp.array([1.0]),
                ]
            )

            du_AB = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    self.alpha_inner,
                    jnp.array([x.alpha_A0]),
                    jnp.array([-1.0]),
                    params.dB * self.sigma_inner,
                    jnp.array([0.0]),  # compatibility
                ]
            )

            rhs_A_inner = jnp.flip(
                (1.0 - params.Kminus * self.dt * c_prev.Z[1:-1]) * c_prev.A[1:-1]
                + params.Kplus * self.dt * c_prev.B[1:-1] * c_prev.Y[1:-1]
            )

            rhs_B_inner = (1.0 - params.Kplus * self.dt * c_prev.Y[1:-1]) * c_prev.B[
                1:-1
            ] + params.Kminus * self.dt * c_prev.A[1:-1] * c_prev.Z[1:-1]

            rhs_AB = jnp.concat(
                [
                    jnp.array([1.0]),
                    rhs_A_inner,
                    jnp.array([0.0, 0.0]),
                    rhs_B_inner,
                    jnp.array([0.0]),
                ]
            )

            current_AB = tridiagonal_solve(dl_AB, d_AB, du_AB, rhs_AB)

            # --- Current Y ---

            rhs_Y_inner = (1.0 - params.Kplus * self.dt * c_prev.B[1:-1]) * c_prev.Y[
                1:-1
            ] + params.Kminus * self.dt * c_prev.A[1:-1] * c_prev.Z[1:-1]
            rhs_Y = jnp.concat([jnp.array([0.0]), rhs_Y_inner, jnp.array([1.0])])

            current_Y = tridiagonal_solve(dl_Y, d_Y, du_Y, rhs_Y)

            # --- Current Z ---

            rhs_Z_inner = (1.0 - params.Kminus * self.dt * c_prev.A[1:-1]) * c_prev.Z[
                1:-1
            ] + params.Kplus * self.dt * c_prev.B[1:-1] * c_prev.Y[1:-1]

            rhs_Z = jnp.concat([jnp.array([0.0]), rhs_Z_inner, jnp.array([0.0])])

            current_Z = tridiagonal_solve(dl_Z, d_Z, du_Z, rhs_Z)

            c = Concentration(
                A=jnp.flip(current_AB[: self.Nx]),
                B=current_AB[self.Nx :],
                Y=current_Y,
                Z=current_Z,
            )

            current = self.compute_current(c, x.k_red, x.k_ox)

            return c, current

        return stepper

    def solve(self, params: SecondOrderECirreMechanismFDMParams) -> Scalar:
        stepper = self._create_stepper(params)

        c_init = Concentration(
            A=jnp.ones_like(self.X),
            B=jnp.zeros_like(self.X),
            Y=jnp.ones_like(self.X),
            Z=jnp.zeros_like(self.X),
        )

        k_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.E0)
        )

        k_ox = params.K0 * jnp.exp(
            (1.0 - params.alpha) * (self.applied_potentials - params.E0)
        )

        alpha_A0 = (
            -self.h1
            * params.K0
            * jnp.exp((1 - params.alpha) * (self.applied_potentials - params.E0))
        )

        alpha_B0 = (
            -self.h1
            * params.K0
            * jnp.exp(-params.alpha * (self.applied_potentials - params.E0))
            / params.dB
        )

        beta_A0 = 1 + self.h1 * params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.E0)
        )

        beta_B0 = (
            1
            + self.h1
            * params.K0
            * jnp.exp((1 - params.alpha) * (self.applied_potentials - params.E0))
            / params.dB
        )

        xs = ScanInputSequence(
            k_red=k_red,
            k_ox=k_ox,
            alpha_A0=alpha_A0,
            alpha_B0=alpha_B0,
            beta_A0=beta_A0,
            beta_B0=beta_B0,
        )

        _, current = scan(stepper, c_init, xs)

        return current
