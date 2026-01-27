from typing import Tuple

from jax.lax import linalg, scan
from jaxtyping import Array, Float

from src.common import compute_current
from src.electrode_kinetics import ButlerVolmerElectrodeKinetics, ButlerVolmerParameters


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return linalg.tridiagonal_solve(dl, d, du, b[:, None]).flatten()


class ImplicitFDSolver1D:
    def __init__(self):
        pass

    def solve(
        self,
        c_init: Float[Array, "space"],
        X: Float[Array, "space"],
        potentials: Float[Array, "time"],
        electrode_kinetics: ButlerVolmerElectrodeKinetics,
        dx: float,
        params: ButlerVolmerParameters,
    ) -> Tuple[Float[Array, "time space"], Float[Array, "time"]]:
        def fdm_stepper(c_prev, theta):
            dl = electrode_kinetics.alpha(X, theta, params)
            d = electrode_kinetics.beta(X, theta, params)
            du = electrode_kinetics.sigma(X, theta, params)
            rhs = electrode_kinetics.delta(c_prev, X, theta, params)

            ck = tridiagonal_solve(dl, d, du, rhs)

            current = compute_current(ck, dx)

            return ck, (ck, current)

        _, (solution, current) = scan(fdm_stepper, c_init, potentials)

        return solution, current
