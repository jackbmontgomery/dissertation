from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import cond, scan
from jaxtyping import Array, ArrayLike, Scalar

from src.params import ElectrodeKineticsParameters
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractFDMSolver, tridiagonal_solve

jax.config.update("jax_enable_x64", True)


class MicroElectrodeFDMSolver(AbstractFDMSolver):
    applied_potentials: Scalar
    dt: float
    n: int
    m: int
    n_e: int
    X: Array
    Y: Array
    y_dl_e: Array
    y_du_e: Array
    y_d_e: Array
    y_dl_nf: Array
    y_du_nf: Array
    y_d_nf: Array
    x_dl: Array
    x_du: Array
    x_d: Array
    vmap_inner_y_sweep: Callable
    vmap_inner_x_sweep: Callable

    def __init__(
        self,
        voltammetry: AbstractVoltammetryTechnique,
        h0: float,
        omega: float,
        dtheta: float,
    ):
        # Suggestion from Understanding Voltammetry 3.4.1
        dt = dtheta / voltammetry.sigma
        self.dt = dt

        T = jnp.linspace(
            voltammetry.t_min,
            voltammetry.t_max,
            int((voltammetry.t_max - voltammetry.t_min) / self.dt),
        )

        self.applied_potentials = vmap(voltammetry.applied_potential)(T)

        x_e = 1.0  # Size of electode in x direction

        x_max = 1.0 + 6.0 * jnp.sqrt(voltammetry.t_max)
        y_max = 6.0 * jnp.sqrt(voltammetry.t_max)

        X, Y, n_e = generate_microband_expanding_mesh(h0, omega, x_e, x_max, y_max)

        self.X = X
        self.Y = Y
        self.n = len(X)
        self.m = len(Y)
        self.n_e = n_e

        y_dl_inner = 2.0 / ((Y[2:] - Y[:-2]) * (Y[1:-1] - Y[:-2]))
        y_du_inner = 2.0 / ((Y[2:] - Y[:-2]) * (Y[2:] - Y[1:-1]))
        y_d_inner = -y_dl_inner - y_du_inner - 2.0 / dt

        # Electode Y-Sweep FDM Matrix
        self.y_dl_e = jnp.concatenate([jnp.array([0.0]), y_dl_inner, jnp.array([0.0])])
        self.y_du_e = jnp.concatenate([jnp.array([0.0]), y_du_inner, jnp.array([0.0])])
        self.y_d_e = jnp.concatenate([jnp.array([1.0]), y_d_inner, jnp.array([1.0])])

        # No Flux Y-Sweep FDM Matrix
        self.y_dl_nf = jnp.concatenate([jnp.array([0.0]), y_dl_inner, jnp.array([0.0])])
        self.y_du_nf = jnp.concatenate([jnp.array([1.0]), y_du_inner, jnp.array([0.0])])
        self.y_d_nf = jnp.concatenate([jnp.array([-1.0]), y_d_inner, jnp.array([1.0])])

        x_dl_inner = 2.0 / ((X[2:] - X[:-2]) * (X[1:-1] - X[:-2]))
        x_du_inner = 2.0 / ((X[2:] - X[:-2]) * (X[2:] - X[1:-1]))

        x_d_inner = -x_dl_inner - x_du_inner - 2.0 / dt

        self.x_dl = jnp.concatenate([jnp.array([0.0]), x_dl_inner, jnp.array([0.0])])
        self.x_du = jnp.concatenate([jnp.array([1.0]), x_du_inner, jnp.array([0.0])])
        self.x_d = jnp.concatenate([jnp.array([-1.0]), x_d_inner, jnp.array([1.0])])

        self.vmap_inner_y_sweep = vmap(self.y_sweep_inner, (0, None, None), 1)
        self.vmap_inner_x_sweep = vmap(self.x_sweep_inner, (0, None), 0)

    def y_sweep_inner(
        self,
        j: int,
        Ck_prev: Array,
        bnd: float,
    ) -> Array:
        l1 = -2.0 / ((self.X[j + 1] - self.X[j - 1]) * (self.X[j] - self.X[j - 1]))
        l3 = -2.0 / ((self.X[j + 1] - self.X[j - 1]) * (self.X[j + 1] - self.X[j]))
        l2 = -l1 - l3 - 2.0 / self.dt

        y_rhs_inner = (
            Ck_prev[1:-1, j - 1] * l1
            + Ck_prev[1:-1, j] * l2
            + Ck_prev[1:-1, j + 1] * l3
        )

        def electode(bnd):
            d0 = bnd
            y_rhs = jnp.concatenate([jnp.array([d0]), y_rhs_inner, jnp.array([1.0])])
            Ck_j = tridiagonal_solve(self.y_dl_e, self.y_d_e, self.y_du_e, y_rhs)
            return Ck_j

        def no_flux(_):
            d0 = 0.0
            y_rhs = jnp.concatenate([jnp.array([d0]), y_rhs_inner, jnp.array([1.0])])
            Ck_j = tridiagonal_solve(self.y_dl_nf, self.y_d_nf, self.y_du_nf, y_rhs)
            return Ck_j

        Ck_j = cond(j < self.n_e, electode, no_flux, bnd)

        return Ck_j

    def x_sweep_inner(self, i: int, Ck_prev: Array) -> Array:
        l1 = -2.0 / ((self.Y[i + 1] - self.Y[i - 1]) * (self.Y[i] - self.Y[i - 1]))
        l3 = -2.0 / ((self.Y[i + 1] - self.Y[i - 1]) * (self.Y[i + 1] - self.Y[i]))
        l2 = -l1 - l3 - 2.0 / self.dt

        x_rhs_inner = (
            Ck_prev[i - 1, 1:-1] * l1
            + Ck_prev[i, 1:-1] * l2
            + Ck_prev[i + 1, 1:-1] * l3
        )

        x_rhs = jnp.concatenate([jnp.array([0.0]), x_rhs_inner, jnp.array([1.0])])

        Ck_i = tridiagonal_solve(self.x_dl, self.x_d, self.x_du, x_rhs)

        return Ck_i

    def y_sweep(self, Ck_prev: Array, bnd: float) -> Array:
        j = jnp.arange(1, self.n - 1)
        Ck_0 = Ck_prev[:, 0]
        Ck_inner = self.vmap_inner_y_sweep(j, Ck_prev, bnd)
        Ck_inf = Ck_prev[:, -1]
        Ck = jnp.concatenate([Ck_0[:, None], Ck_inner, Ck_inf[:, None]], axis=1)
        return Ck

    def x_sweep(self, c_prev: Array) -> Array:
        i = jnp.arange(1, self.m - 1)
        c_0 = c_prev[0]
        c_inner = self.vmap_inner_x_sweep(i, c_prev)
        c_inf = c_prev[-1]
        c = jnp.concatenate([c_0[None, :], c_inner, c_inf[None, :]], axis=0)
        return c

    def compute_current(self, ck: Array) -> Array:
        J1 = ck[1, : self.n_e - 1] - ck[0, : self.n_e - 1]
        J2 = ck[1, 1 : self.n_e] - ck[0, 1 : self.n_e]

        current_electrode = (
            (-0.5 / (self.Y[1] - self.Y[0]))
            * (J2 + J1)
            * (self.X[1 : self.n_e] - self.X[: self.n_e - 1])
        )
        current = jnp.sum(current_electrode)
        return current

    def solve(self, params: ElectrodeKineticsParameters) -> Scalar:
        def fdm_stepper(Ck, bnd) -> Tuple[Array, ArrayLike]:
            Ck = self.y_sweep(Ck, bnd)
            J = self.compute_current(Ck)
            Ck = self.x_sweep(Ck)
            return Ck, J

        c_init = jnp.ones((self.m, self.n))
        # NOTE: This needs to change but is here for now to test
        bnds = 1 / (1 + jnp.exp(-self.applied_potentials))
        _final_c, current = scan(fdm_stepper, c_init, bnds)
        return current


def generate_microband_expanding_mesh(
    h0: float, omega: float, x_e: float, x_max: ArrayLike, y_max: ArrayLike
):
    Y = []
    h = h0
    Y.append(0.0)
    while Y[-1] <= y_max:
        Y.append(Y[-1] + h)
        h *= omega

    Y = jnp.array(Y)

    X = []
    h = h0
    X.append(0.0)

    while X[-1] < x_e / 2.0:
        X.append(X[-1] + h)
        h *= omega

    X[-1] = x_e / 2.0

    for i in range(len(X) - 2, -1, -1):
        X.append(1 - X[i])

    n_e = len(X)
    h = h0
    while X[-1] <= x_max:
        X.append(X[-1] + h)
        h *= omega

    X = jnp.array(X)

    return X, Y, n_e
