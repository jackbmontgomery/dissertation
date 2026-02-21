import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import SecondOrderECirreMechanismFDMParams
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver


@dataclass
class Concentration:
    A: Scalar
    B: Scalar
    Y: Scalar
    Z: Scalar


class SecondOrderECirreFDMSolverExplicitApprox(AbstractFDMSolver):
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

    def newton_solve(
        self,
        c: Concentration,
        params: SecondOrderECirreMechanismFDMParams,
        c_prev: Concentration,
        applied_potential: Scalar,
    ) -> Concentration:
        k_red = params.K0 * jnp.exp(-params.alpha * (applied_potential - params.E0))
        k_ox = params.K0 * jnp.exp(
            (1.0 - params.alpha) * (applied_potential - params.E0)
        )

        # Electrode Boundary Conditions
        f0_A = c.A[0] * (1 + self.h1 * k_red) - c.B[0] * self.h1 * k_ox - c.A[1]
        f0_B = (
            -c.A[0] * self.h1 * k_red / params.dB
            - c.B[0] * (1 + self.h1 * k_ox / params.dB)
            - c.A[1]
        ) - c.B[1]
        f0_Y = c.Y[1] - c.Y[0]
        f0_Z = c.Z[1] - c.Z[0]

        # Inner conditions
        gamma = (
            -params.Kplus * self.dt * c.B[1:-1] * c.Y[1:-1]
            + params.Kminus * self.dt * c.A[1:-1] * c.Z[1:-1]
        )

        fi_A = (
            self.alpha_inner * c.A[:-2]
            + ((1 - (self.alpha_inner + self.sigma_inner)) * c.A[1:-1])
            + self.sigma_inner * c.A[2:]
            + gamma
            - c_prev.A[1:-1]
        )

        fi_B = (
            params.dB * self.alpha_inner * c.B[:-2]
            + ((1 - params.dB * (self.alpha_inner + self.sigma_inner)) * c.B[1:-1])
            + params.dB * self.sigma_inner * c.B[2:]
            - gamma
            - c_prev.B[1:-1]
        )

        fi_Y = (
            params.dY * self.alpha_inner * c.Y[:-2]
            + ((1 - params.dY * (self.alpha_inner + self.sigma_inner)) * c.Y[1:-1])
            + params.dY * self.sigma_inner * c.Y[2:]
            - gamma
            - c_prev.Y[1:-1]
        )

        fi_Z = (
            params.dZ * self.alpha_inner * c.Z[:-2]
            + ((1 - params.dZ * (self.alpha_inner + self.sigma_inner)) * c.Z[1:-1])
            + params.dZ * self.sigma_inner * c.Z[2:]
            - gamma
            - c_prev.Z[1:-1]
        )

        # Semi-infinite Boundary Condition (Actually N-1, but that does not make nice variable name)
        fn_A = c.A[-1] - 1.0
        fn_B = c.B[-1]
        fn_Y = c.Y[-1] - 1.0
        fn_Z = c.Z[-1]

        F = jnp.concat(
            [f0_A, f0_B, f0_Y, f0_Z, fi_A, fi_B, fi_Y, fi_Z, fn_A, fn_B, fn_Y, fn_Z]
        )

        # Jacobian
        return None
