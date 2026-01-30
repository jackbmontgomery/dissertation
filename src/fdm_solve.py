from equinox import filter_jit
from jax.lax import linalg, scan
from jaxtyping import Array, Scalar

from src.fdm_discretisation import AbstractFDMDiscretisation
from src.pde_parameters import AbstractPDEParameters


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return linalg.tridiagonal_solve(dl, d, du, b[:, None]).flatten()


@filter_jit
def fdm_implicit_solve(
    init: Scalar,
    forcing: Scalar,
    fdm_discretisation: AbstractFDMDiscretisation,
    fdm_params: AbstractPDEParameters,
):
    def fdm_stepper(carry: Array, forcing: Scalar):
        (dl, d, du), b = fdm_discretisation.operator_and_vector(
            carry, forcing, fdm_params
        )

        carry = tridiagonal_solve(dl, d, du, b)

        return carry, carry

    _, fdm_solution = scan(fdm_stepper, init, forcing)

    return fdm_solution
