import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Scalar

from src.params import SecondOrderECirreMechanismFDMParams
from src.solvers import nonadiagonal_solve
from src.utils import interleave_concat_4d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver


@dataclass
class ScanInputSequence:
    k_red: Scalar
    k_ox: Scalar


@dataclass
class Concentration:
    A: Scalar
    B: Scalar
    Y: Scalar
    Z: Scalar


class SecondOrderECirreFDMSolverBackwardImplicit(AbstractFDSolver):
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
        dl4_inner = interleave_concat_4d(
            self.alpha_inner,
            params.dB * self.alpha_inner,
            params.dY * self.alpha_inner,
            params.dZ * self.alpha_inner,
        )

        dl4 = jnp.concat([jnp.zeros((4,)), dl4_inner, jnp.zeros((4,))])

        du4_inner = interleave_concat_4d(
            self.sigma_inner,
            params.dB * self.sigma_inner,
            params.dY * self.sigma_inner,
            params.dZ * self.sigma_inner,
        )

        du4 = jnp.concat(
            [jnp.array([-1.0, -1.0, 1.0, 1.0]), du4_inner, jnp.zeros((4,))]
        )

        def stepper(c_prev: Concentration, x: ScanInputSequence):
            dl3_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                params.Kminus * self.dt * c_prev.Z[1:-1],
            )

            dl3 = jnp.concat([jnp.zeros((4,)), dl3_inner, jnp.zeros((4,))])

            dl2_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                jnp.zeros_like(self.alpha_inner),
                -params.Kminus * self.dt * c_prev.Z[1:-1],
                -params.Kplus * self.dt * c_prev.Y[1:-1],
            )

            dl2 = jnp.concat([jnp.zeros((4,)), dl2_inner, jnp.zeros((4,))])

            dl1_inner = interleave_concat_4d(
                jnp.zeros_like(self.alpha_inner),
                -params.Kminus * self.dt * c_prev.Z[1:-1],
                params.Kplus * self.dt * c_prev.Y[1:-1],
                -params.Kplus * self.dt * c_prev.B[1:-1],
            )

            dl1 = jnp.concat(
                [
                    jnp.array([0.0, -self.h1 * x.k_red / params.dB, 0.0, 0.0]),
                    dl1_inner,
                    jnp.zeros((4,)),
                ]
            )

            d_first = jnp.array(
                [1 + self.h1 * x.k_red, 1 + self.h1 * x.k_ox / params.dB, -1.0, -1.0]
            )

            d_A_inner = (
                1.0
                - (self.alpha_inner + self.sigma_inner)
                + params.Kminus * self.dt * c_prev.Z[1:-1]
            )

            d_B_inner = (
                1.0
                - params.dB * (self.alpha_inner + self.sigma_inner)
                + params.Kplus * self.dt * c_prev.Y[1:-1]
            )

            d_Y_inner = (
                1.0
                - params.dY * (self.alpha_inner + self.sigma_inner)
                + params.Kplus * self.dt * c_prev.B[1:-1]
            )

            d_Z_inner = (
                1.0
                - params.dZ * (self.alpha_inner + self.sigma_inner)
                + params.Kminus * self.dt * c_prev.A[1:-1]
            )

            d_inner = interleave_concat_4d(d_A_inner, d_B_inner, d_Y_inner, d_Z_inner)

            d_last = jnp.full((4,), 1.0)

            d = jnp.concat([d_first, d_inner, d_last])

            du1_inner = interleave_concat_4d(
                -params.Kplus * self.dt * c_prev.Y[1:-1],
                params.Kplus * self.dt * c_prev.B[1:-1],
                -params.Kminus * self.dt * c_prev.A[1:-1],
                jnp.zeros_like(self.sigma_inner),
            )

            du1 = jnp.concat(
                [
                    jnp.array([-self.h1 * x.k_ox, 0.0, 0.0, 0.0]),
                    du1_inner,
                    jnp.zeros((4,)),
                ]
            )

            du2_inner = interleave_concat_4d(
                -params.Kplus * self.dt * c_prev.B[1:-1],
                -params.Kminus * self.dt * c_prev.A[1:-1],
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
            )

            du2 = jnp.concat(
                [
                    jnp.zeros((4,)),
                    du2_inner,
                    jnp.zeros((4,)),
                ]
            )

            du3_inner = interleave_concat_4d(
                params.Kminus * self.dt * c_prev.A[1:-1],
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
                jnp.zeros_like(self.sigma_inner),
            )

            du3 = jnp.concat(
                [
                    jnp.zeros((4,)),
                    du3_inner,
                    jnp.zeros((4,)),
                ]
            )

            rhs_A_inner = (1.0 + params.Kminus * self.dt * c_prev.Z[1:-1]) * c_prev.A[
                1:-1
            ] - params.Kplus * self.dt * c_prev.B[1:-1] * c_prev.Y[1:-1]

            rhs_B_inner = (1.0 + params.Kplus * self.dt * c_prev.Y[1:-1]) * c_prev.B[
                1:-1
            ] - params.Kminus * self.dt * c_prev.A[1:-1] * c_prev.Z[1:-1]

            rhs_Y_inner = (1.0 + params.Kplus * self.dt * c_prev.B[1:-1]) * c_prev.Y[
                1:-1
            ] - params.Kminus * self.dt * c_prev.A[1:-1] * c_prev.Z[1:-1]

            rhs_Z_inner = (1.0 + params.Kminus * self.dt * c_prev.A[1:-1]) * c_prev.Z[
                1:-1
            ] - params.Kplus * self.dt * c_prev.B[1:-1] * c_prev.Y[1:-1]

            rhs_inner = interleave_concat_4d(
                rhs_A_inner, rhs_B_inner, rhs_Y_inner, rhs_Z_inner
            )

            rhs = jnp.concat(
                [jnp.zeros((4,)), rhs_inner, jnp.array([1.0, 0.0, 1.0, 0.0])]
            )

            # print(
            #     dl4.shape,
            #     dl3.shape,
            #     dl2.shape,
            #     dl1.shape,
            #     d.shape,
            #     du1.shape,
            #     du2.shape,
            #     du3.shape,
            #     du4.shape,
            #     rhs.shape,
            # )
            c_array = nonadiagonal_solve(dl4, dl3, dl2, dl1, d, du1, du2, du3, du4, rhs)

            c = Concentration(
                A=c_array[::4], B=c_array[1::4], Y=c_array[2::4], Z=c_array[3::4]
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

        xs = ScanInputSequence(k_red=k_red, k_ox=k_ox)

        _, current = scan(stepper, c_init, xs)

        return current
