from typing import Callable

from jax import jit
from jaxtyping import Array, Scalar

from src.fdm_discretisation import AbstractFDMDiscretisation, fdm_implicit_solve
from src.pde_parameters import AbstractPDEParameters


def compute_solution_current(solution: Array, dx: float) -> Array:
    c0 = solution[:, 0]
    c1 = solution[:, 1]
    c2 = solution[:, 2]
    return -(-c2 + 4.0 * c1 - 3.0 * c0) / (2.0 * dx)


def create_fdm_current_simulator(
    c_init: Scalar,
    forcing: Scalar,
    fdm_discretisation: AbstractFDMDiscretisation,
    dx: float,
) -> Callable[[AbstractPDEParameters], Scalar]:
    @jit
    def simulate(
        params: AbstractPDEParameters,
    ):
        fdm_solution = fdm_implicit_solve(
            init=c_init,
            forcing=forcing,
            fdm_discretisation=fdm_discretisation,
            fdm_params=params,
        )
        current = compute_solution_current(fdm_solution, dx=dx)
        return current

    return simulate
