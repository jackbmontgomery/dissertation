from typing import Callable, Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import ECirreMechanismFDMParams
from src.solvers import pentadiagonal_solve
from src.utils import interleave_concat_2d
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDSolver


@dataclass
class ScanInputSequence:
    beta_A0: Scalar
    beta_B0: Scalar
    eta_A0: Scalar
    eta_B0: Scalar


class ECirreMechanismFDMSolver(AbstractFDSolver):
    applied_potentials: Scalar
    dt: float
    X: Array
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

        print("Discretisation", f"X: {X.shape}", f"T: {T.shape}")
        self.X = X
        self.Nx = len(X)

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h1 = X[1] - X[0]
        self.h2 = X[2] - X[0]
        self.alpha_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.sigma_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))

    def compute_current(self, c: Scalar) -> Scalar:
        c0 = c[0]
        c1 = c[2]
        c2 = c[4]

        dcdx = (self.h2**2 * (c0 - c1) + self.h1**2 * (c2 - c0)) / (
            self.h1 * self.h2 * (self.h1 - self.h2)
        )

        return -dcdx

    def _create_stepper(
        self,
        params: ECirreMechanismFDMParams,
    ) -> Callable[
        [Scalar, ScanInputSequence],
        Tuple[Scalar, Scalar],
    ]:
        d2l_inner = interleave_concat_2d(self.alpha_inner, params.dB * self.alpha_inner)

        d2l = jnp.concat(
            [
                jnp.array([0.0, 0.0]),
                d2l_inner,
                jnp.array([0.0, 0.0]),
            ]
        )

        dlB_inner = jnp.zeros((self.Nx - 2,))
        dlA_inner = jnp.full((self.Nx - 2,), -params.Kminus * self.dt)
        dl_inner = interleave_concat_2d(dlB_inner, dlA_inner)

        dA_inner = 1 - (self.alpha_inner + self.sigma_inner) + params.Kminus * self.dt

        dB_inner = (
            1
            - params.dB * (self.alpha_inner + self.sigma_inner)
            + params.Kplus * self.dt
        )

        d_inner = interleave_concat_2d(dA_inner, dB_inner)

        duB_inner = jnp.full((self.Nx - 2,), -params.Kplus * self.dt)
        duA_inner = jnp.zeros((self.Nx - 2,))
        du_inner = interleave_concat_2d(duB_inner, duA_inner)

        d2u_inner = interleave_concat_2d(
            self.sigma_inner,
            params.dB * self.sigma_inner,
        )

        d2u = jnp.concatenate(
            [jnp.array([-1.0, -1.0]), d2u_inner, jnp.array([0.0, 0.0])]
        )

        def stepper(c_prev: Scalar, x: ScanInputSequence) -> Tuple[Scalar, Scalar]:
            dl = jnp.concat(
                [
                    jnp.array([0.0, x.eta_A0]),
                    dl_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            d = jnp.concat(
                [
                    jnp.array([x.beta_A0, x.beta_B0]),
                    d_inner,
                    jnp.array([1.0, 1.0]),
                ]
            )

            du = jnp.concat(
                [
                    jnp.array([x.eta_B0, 0.0]),
                    du_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            rhs = jnp.concatenate(
                [
                    jnp.array([0.0, 0.0]),
                    c_prev[2:-2],
                    jnp.array([1.0, 0.0]),
                ]
            )

            Ck = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

            current = self.compute_current(Ck)

            return Ck, current

        return stepper

    def solve(self, params: ECirreMechanismFDMParams) -> Scalar:
        stepper = self._create_stepper(params)

        c_init = interleave_concat_2d(jnp.ones_like(self.X), jnp.zeros_like(self.X))

        k_red = params.K0 * jnp.exp(
            -params.alpha * (self.applied_potentials - params.E0)
        )
        k_ox = params.K0 * jnp.exp(
            (1.0 - params.alpha) * (self.applied_potentials - params.E0)
        )

        beta_A0 = 1.0 + self.h1 * k_red
        beta_B0 = 1.0 + self.h1 * k_ox / params.dB

        eta_A0 = -self.h1 * k_red / params.dB
        eta_B0 = -self.h1 * k_ox

        xs = ScanInputSequence(
            beta_A0=beta_A0,
            beta_B0=beta_B0,
            eta_A0=eta_A0,
            eta_B0=eta_B0,
        )

        _, current = scan(stepper, c_init, xs)
        return current
