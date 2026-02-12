from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from jax import vmap
from jaxtyping import PRNGKeyArray, Scalar


def generate_noisy_samples(
    num_samples: int,
    current: Scalar,
    sigma: float,
    *,
    key: PRNGKeyArray,
):
    def add_noise(r_key: PRNGKeyArray, current=current):
        noisy_current = current + jr.normal(r_key, shape=current.shape) * sigma
        return noisy_current

    keys = jr.split(key, num_samples)
    samples = vmap(add_noise)(keys)
    return samples


def interleave_concat(a, b):
    return jnp.column_stack([a, b]).reshape(-1)


def bfgs_minimise(initial_parameters, log_density: Callable):
    bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
    solution = optx.minimise(
        lambda params, _: -log_density(params), bfgs, initial_parameters
    )
    sampling_init_params = solution.value
    return sampling_init_params
