from typing import Callable

from jax import jit
from jaxtyping import Array, Scalar

from src.fdm_dep import AbstractFDMSolver
from src.fdm_discretisation import AbstractFDMDiscretisation, fdm_implicit_solve
from src.pde_parameters import AbstractPDEParameters


def compute_solution_current(solution: Array, X: Scalar) -> Array:
    c0 = solution[:, 0]
    c1 = solution[:, 1]
    c2 = solution[:, 2]

    h1 = X[1] - X[0]
    h2 = X[2] - X[0]

    dcdx = (h2**2 * (c0 - c1) + h1**2 * (c2 - c0)) / (h1 * h2 * (h1 - h2))

    return -dcdx


def create_current_simulator(
    fdm_solver: AbstractFDMSolver,
) -> Callable[[AbstractPDEParameters], Scalar]:
    @jit
    def simulate(
        params: AbstractPDEParameters,
    ):
        fdm_solution = fdm_solver.solve(params)
        current = compute_solution_current(solution)

        return current

    return simulate
