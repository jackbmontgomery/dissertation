import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from jaxtyping import PRNGKeyArray

from src.fdm import AbstractFDMSolver
from src.params import ElectrodeKineticsParameters


def generate_noisy_samples(
    num_samples: int,
    params: ElectrodeKineticsParameters,
    sigma: float,
    fdm_solver: AbstractFDMSolver,
    *,
    key: PRNGKeyArray,
):
    current = fdm_solver.solve(params)

    def add_noise(r_key: PRNGKeyArray, current=current):
        noisy_current = current + jr.normal(r_key, shape=current.shape) * sigma
        return noisy_current

    keys = jr.split(key, num_samples)
    samples = vmap(add_noise)(keys)
    return samples


def interleave_concat(a, b):
    return jnp.column_stack([a, b]).reshape(-1)
