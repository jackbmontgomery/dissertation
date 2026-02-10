from typing import Callable

import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jaxtyping import Array, Scalar

from src.params import LinearECIrreversibleParams
from src.utils import interleave_concat
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver


class LinearECIrreversibleFDMSolver(AbstractFDMSolver):
    applied_potentials: Scalar
    dt: float
    X: Array
    h: Scalar
    dl_A_inner: Scalar
    d_A_inner: Scalar
    du_A_inner: Scalar

    def __init__(
        self, voltammetry: AbstractVoltammetryTechnique, h: float, dtheta: float
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

        X_plus = X[1:-1] - X[:-2]
        X_minus = X[2:] - X[1:-1]

        self.h = X[1] - X[0]
        self.dl_A_inner = -(2.0 * dt) / (X_minus * (X_minus + X_plus))
        self.du_A_inner = -(2.0 * dt) / (X_plus * (X_minus + X_plus))
        self.d_A_inner = 1 - self.dl_A_inner - self.du_A_inner

    def compute_current(self, ck: Array) -> Array:
        # Only for A
        c0 = ck[0]
        c1 = ck[2]
        c2 = ck[4]

        h1 = self.X[1] - self.X[0]
        h2 = self.X[2] - self.X[0]

        dcdx = (h2**2 * (c0 - c1) + h1**2 * (c2 - c0)) / (h1 * h2 * (h1 - h2))

        return -dcdx

    def _create_stepper(
        self,
        params: LinearECIrreversibleParams,
    ) -> Callable[[Array, Scalar], Array]:
        def stepper(c_prev: Array, applied_potential: Scalar):
            k_red = params.kappa * jnp.exp(
                -params.alpha * (applied_potential - params.epsilon)
            )
            k_ox = params.kappa * jnp.exp(
                (1.0 - params.alpha) * (applied_potential - params.epsilon)
            )

            # Main diagonal: A0, B0, A1, B1, ...
            dA0 = jnp.array([1.0 + self.h + k_red])
            dB0 = jnp.array([1.0 + self.h + k_ox / params.deltab])

            dA_inner = self.d_A_inner + params.kappam * self.dt
            dB_inner = params.deltab * self.d_A_inner + params.kappap * self.dt

            d_inner = interleave_concat(dA_inner, dB_inner)

            dAf = jnp.array([1.0])
            dBf = jnp.array([1.0])

            d = jnp.concat([dA0, dB0, d_inner, dAf, dBf])

            # NOTE: I want to change the inner here but not sure how yet. Maybe generic inner in the class that I multiply by params.km...

            # Lower diagonal: A_coef, B0, B0, A1, B1, ...

            dlB_inner = jnp.zeros((len(self.X) - 2,))
            dlA_inner = jnp.full((len(self.X) - 2,), -1 * params.kappam * self.dt)
            dl_inner = interleave_concat(dlB_inner, dlA_inner)

            dl = jnp.concat(
                [
                    jnp.array([-self.h * k_red / params.deltab]),
                    dl_inner,
                    jnp.array([0.0, 0.0]),
                ]
            )

            # 2nd Lower diagonal: A1, B1, A2, B2, ..., AN, BN

            d2lA = jnp.concatenate(
                [
                    self.dl_A_inner,
                    jnp.array([0.0]),
                ]
            )

            d2lB = jnp.concatenate(
                [
                    params.deltab * self.dl_A_inner,
                    jnp.array([0.0]),
                ]
            )

            d2l = interleave_concat(d2lA, d2lB)

            # Upper diagonal: B_coef, A0, B0, A1, B1, ...

            duB_inner = jnp.full((len(self.X) - 2,), -1 * params.kappap * self.dt)
            duA_inner = jnp.zeros((len(self.X) - 2,))
            du_inner = interleave_concat(duB_inner, duA_inner)

            du = jnp.concat(
                [
                    jnp.array([-self.h * k_ox, 0.0]),
                    du_inner,
                    jnp.array([0.0]),
                ]
            )

            # Second Upper Diagonal: A0, B0, ..., A(N-1), B(N-1)

            d2uA = jnp.concatenate(
                [
                    jnp.array([-1.0]),
                    self.du_A_inner,
                ]
            )

            d2uB = jnp.concatenate(
                [
                    jnp.array([-1.0]),
                    params.deltab * self.du_A_inner,
                ]
            )

            d2u = interleave_concat(d2uA, d2uB)

            A = (
                jnp.diag(d)
                + jnp.diag(dl, k=-1)
                + jnp.diag(d2l, k=-2)
                + jnp.diag(du, k=1)
                + jnp.diag(d2u, k=2)
            )

            b = jnp.concatenate(
                [
                    jnp.array([0.0]),
                    jnp.array([0.0]),
                    c_prev[2:-2],
                    jnp.array([1.0]),
                    jnp.array([0.0]),
                ]
            )

            c = jnp.linalg.solve(A, b)

            current = self.compute_current(c)

            return c, current

        return stepper

    def solve(self, params: LinearECIrreversibleParams) -> Scalar:
        c_init = interleave_concat(jnp.ones_like(self.X), jnp.zeros_like(self.X))
        stepper = self._create_stepper(params)
        _, current = scan(stepper, c_init, self.applied_potentials)
        return current
