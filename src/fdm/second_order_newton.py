from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optimistix as optx
from chex import dataclass
from jax import vmap
from jax.lax import scan, while_loop
from jaxtyping import Scalar

from src.params import SecondOrderECirreMechanismFDMParams
from src.solvers import nonadiagonal_solve
from src.utils import interleave_concat_4d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver


@dataclass
class Concentration:
    A: Scalar
    B: Scalar
    Y: Scalar
    Z: Scalar


@dataclass
class WhileOpArgs:
    c: Concentration
    delta_c: Scalar


@dataclass
class ScanInputSequence:
    applied_potential: Scalar


class SecondOrderECirreFDMSolverNewton(AbstractFDMSolver):
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

    def create_build_J_diags(
        self,
        params: SecondOrderECirreMechanismFDMParams,
        c_prev: Concentration,
        k_red: Scalar,
        k_ox: Scalar,
    ) -> Callable:
        def build_J(
            c: Concentration,
        ) -> Tuple[
            Scalar,
            Scalar,
            Scalar,
            Scalar,
            Scalar,
            Scalar,
            Scalar,
            Scalar,
            Scalar,
        ]:
            km_a = params.Kminus * self.dt * c.A[1:-1]
            kp_b = params.Kplus * self.dt * c.B[1:-1]
            kp_y = params.Kplus * self.dt * c.Y[1:-1]
            km_z = params.Kminus * self.dt * c.Z[1:-1]

            # Lower Diagonals

            alpha_A_inner = self.alpha_inner
            alpha_B_inner = params.dB * self.alpha_inner
            alpha_Y_inner = params.dY * self.alpha_inner
            alpha_Z_inner = params.dZ * self.alpha_inner

            dl4_inner = interleave_concat_4d(
                alpha_A_inner, alpha_B_inner, alpha_Y_inner, alpha_Z_inner
            )

            dl4 = jnp.concat(
                [
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                    dl4_inner,
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                ]
            )

            dl3_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                km_z,
            )

            dl3 = jnp.concat(
                [
                    jnp.array([0.0, 0.0, 0.0]),
                    jnp.array([0.0]),
                    dl3_inner,
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                ]
            )

            dl2_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                -km_z,
                -kp_y,
            )

            dl2 = jnp.concat(
                [
                    jnp.array([0.0, 0.0]),
                    jnp.array([0.0, 0.0]),
                    dl2_inner,
                    jnp.zeros((4,)),
                ]
            )

            dl1_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                -km_z,
                kp_y,
                -kp_b,
            )

            dl1 = jnp.concat(
                [
                    jnp.array([0.0]),
                    jnp.array([-self.h1 * k_red / params.dB, 0.0, 0.0]),
                    dl1_inner,
                    jnp.zeros((4,)),
                ]
            )

            # Main Diagonal
            d_A_inner = 1 - (self.alpha_inner + self.sigma_inner) + km_z
            d_B_inner = 1 - params.dB * (self.alpha_inner + self.sigma_inner) + kp_y
            d_Y_inner = 1 - params.dY * (self.alpha_inner + self.sigma_inner) + kp_b
            d_Z_inner = 1 - params.dZ * (self.alpha_inner + self.sigma_inner) + km_a

            d_inner = interleave_concat_4d(d_A_inner, d_B_inner, d_Y_inner, d_Z_inner)

            d = jnp.concat(
                [
                    jnp.array(
                        [
                            1 + self.h1 * k_red,
                            1 + self.h1 * k_ox / params.dB,
                            -1.0,
                            -1.0,
                        ]
                    ),
                    d_inner,
                    jnp.ones((4,)),
                ]
            )

            # Upper Diagonals

            du1_inner = interleave_concat_4d(
                -kp_y, kp_b, -km_a, jnp.zeros_like(self.sigma_inner)
            )

            du1 = jnp.concat(
                [
                    jnp.array([-self.h1 * k_ox, 0.0, 0.0, 0.0]),
                    du1_inner,
                    jnp.array([0.0, 0.0, 0.0]),
                    jnp.array([0.0]),
                ]
            )

            du2_inner = interleave_concat_4d(
                -kp_b,
                -km_a,
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
            )

            du2 = jnp.concat(
                [
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                    du2_inner,
                    jnp.array([0.0, 0.0]),
                    jnp.array([0.0, 0.0]),
                ]
            )

            du3_inner = interleave_concat_4d(
                km_a,
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
            )

            du3 = jnp.concat(
                [
                    jnp.array([0.0, 0.0, 0.0, 0.0]),
                    du3_inner,
                    jnp.array([0.0]),
                    jnp.array([0.0, 0.0, 0.0]),
                ]
            )

            du4_A_inner = self.sigma_inner
            du4_B_inner = params.dB * self.sigma_inner
            du4_Y_inner = params.dY * self.sigma_inner
            du4_Z_inner = params.dZ * self.sigma_inner

            du4_inner = interleave_concat_4d(
                du4_A_inner, du4_B_inner, du4_Y_inner, du4_Z_inner
            )

            du4 = jnp.concat(
                [jnp.array([-1.0, -1.0, 1.0, 1.0]), du4_inner, jnp.zeros((4,))]
            )

            return dl4, dl3, dl2, dl1, d, du1, du2, du3, du4

        return build_J

    def create_build_F(
        self,
        params: SecondOrderECirreMechanismFDMParams,
        c_prev: Concentration,
        k_red: Scalar,
        k_ox: Scalar,
    ) -> Callable[[Concentration], Scalar]:
        def build_F(c: Concentration):
            # Electrode Boundary Conditions
            f0_A = c.A[0] * (1 + self.h1 * k_red) - c.B[0] * self.h1 * k_ox - c.A[1]

            f0_B = (
                -c.A[0] * self.h1 * k_red / params.dB
                + c.B[0] * (1 + self.h1 * k_ox / params.dB)
                - c.B[1]
            )

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
                + gamma
                - c_prev.Z[1:-1]
            )

            # Semi-infinite Boundary Condition (Actually N-1, but that does not make nice variable name)
            fn_A = c.A[-1] - 1.0
            fn_B = c.B[-1]
            fn_Y = c.Y[-1] - 1.0
            fn_Z = c.Z[-1]

            F_inner = interleave_concat_4d(fi_A, fi_B, fi_Y, fi_Z)

            F = jnp.concat(
                [
                    jnp.array([f0_A, f0_B, f0_Y, f0_Z]),
                    F_inner,
                    jnp.array([fn_A, fn_B, fn_Y, fn_Z]),
                ]
            )

            return F

        return build_F

    def newton_solve(
        self,
        c_prev: Concentration,
        params: SecondOrderECirreMechanismFDMParams,
        k_red: Scalar,
        k_ox: Scalar,
    ):
        a_tol = 1e-3

        build_F = self.create_build_F(params, c_prev, k_red, k_ox)
        build_J_diags = self.create_build_J_diags(params, c_prev, k_red, k_ox)

        def cond_fun(x: WhileOpArgs):
            return jnp.less_equal(a_tol, jnp.max(jnp.abs(x.delta_c)))

        def body_fun(x: WhileOpArgs):
            F = build_F(x.c)
            dl4, dl3, dl2, dl1, d, du1, du2, du3, du4 = build_J_diags(x.c)

            delta_c = nonadiagonal_solve(dl4, dl3, dl2, dl1, d, du1, du2, du3, du4, -F)

            c_A = x.c.A + delta_c[::4]
            c_B = x.c.B + delta_c[1::4]
            c_Y = x.c.Y + delta_c[2::4]
            c_Z = x.c.Z + delta_c[3::4]

            c = Concentration(A=c_A, B=c_B, Y=c_Y, Z=c_Z)

            new_x = WhileOpArgs(c=c, delta_c=delta_c)
            return new_x

        x = WhileOpArgs(c=c_prev, delta_c=jnp.ones((4 * self.Nx,)))

        x = body_fun(x)

        return x.c

    def create_stepper(self, params: SecondOrderECirreMechanismFDMParams) -> Callable:
        def stepper(c_prev: Concentration, x: ScanInputSequence):
            k_red = params.K0 * jnp.exp(
                -params.alpha * (x.applied_potential - params.E0)
            )
            k_ox = params.K0 * jnp.exp(
                (1.0 - params.alpha) * (x.applied_potential - params.E0)
            )

            c = self.newton_solve(c_prev, params, k_red, k_ox)
            current = self.compute_current(c, k_red, k_ox)
            return c, current

        return stepper

    def solve(self, params: SecondOrderECirreMechanismFDMParams) -> Scalar:
        stepper = self.create_stepper(params)
        init_c = Concentration(
            A=jnp.ones_like(self.X),
            B=jnp.zeros_like(self.X),
            Y=jnp.ones_like(self.X),
            Z=jnp.zeros_like(self.X),
        )
        xs = ScanInputSequence(applied_potential=self.applied_potentials)
        _, current = scan(stepper, init_c, xs)

        return current
