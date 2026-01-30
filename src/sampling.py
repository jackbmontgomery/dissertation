from typing import Callable, NamedTuple, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, PyTree

from src.pde_parameters import AbstractPDEParameters


def generate_noisy_samples(
    num_samples: int,
    simulate: Callable[[AbstractPDEParameters], Array],
    params: AbstractPDEParameters,
    sigma: float,
    *,
    key: PRNGKeyArray,
):
    base = simulate(params)

    def add_noise(r_key: PRNGKeyArray, base=base):
        noisy_current = base + jr.normal(r_key, shape=base.shape) * sigma
        return noisy_current

    keys = jr.split(key, num_samples)
    samples = vmap(add_noise)(keys)
    return samples


def _inference_loop(
    key: PRNGKeyArray,
    kernel: Callable[[PRNGKeyArray, AbstractPDEParameters], Tuple[PyTree, PyTree]],
    initial_state: NamedTuple,
    num_samples: int,
):
    @jit
    def scan_step(state, step_key):
        state, info = kernel(step_key, state)
        return state, state

    keys = jr.split(key, num_samples)
    _, states = scan(scan_step, initial_state, keys)

    return states


def nuts_sampling(
    key: PRNGKeyArray,
    n_samples: int,
    initial_parameters: AbstractPDEParameters,
    log_density: Callable[[AbstractPDEParameters, Array], Array],
    step_size: float = 1e-3,
    inverse_mass_matrix: Array = jnp.array([0.1, 0.1]),
):
    nuts = blackjax.nuts(log_density, inverse_mass_matrix)

    initial_state = nuts.init(initial_parameters, key)

    kernel = jit(nuts.step)

    states = _inference_loop(key, kernel, initial_state, n_samples)

    return states


def rw_sampling(
    key: PRNGKeyArray,
    n_samples: int,
    initial_parameters: AbstractPDEParameters,
    log_density: Callable[[AbstractPDEParameters, Array], Array],
    sigma: Array = jnp.array([0.1, 0.1]),
):
    rw = blackjax.additive_step_random_walk(
        log_density, blackjax.mcmc.random_walk.normal(sigma)
    )

    initial_state = rw.init(initial_parameters, key)

    kernel = jit(rw.step)

    states = _inference_loop(key, kernel, initial_state, n_samples)

    return states
