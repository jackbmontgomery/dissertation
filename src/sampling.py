from typing import Callable, NamedTuple, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, pmap, vmap
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
        return state, (state, info)

    keys = jr.split(key, num_samples)
    _, (states, infos) = scan(scan_step, initial_state, keys)

    return states, infos


inference_loop_multiple_chains: Callable = pmap(
    _inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
)


def rw_sampling(
    key: PRNGKeyArray,
    n_samples: int,
    initial_parameters: AbstractPDEParameters,
    log_density: Callable[[AbstractPDEParameters, Array], Array],
    sigma: Array = jnp.array([0.01, 0.01, 0.01]),
):
    rw = blackjax.additive_step_random_walk(
        log_density, blackjax.mcmc.random_walk.normal(sigma)
    )

    initial_state = rw.init(initial_parameters, key)

    jit_step = jit(rw.step)

    states, infos = _inference_loop(key, jit_step, initial_state, n_samples)

    return states, infos


def mclmc_sampling(
    keys: PRNGKeyArray,
    n_samples_per_chain: int,
    initial_parameters: AbstractPDEParameters,
    log_density: Callable[[AbstractPDEParameters, Array], Array],
    step_size: float = 1e-2,
):
    mclmc = blackjax.adjusted_mclmc_dynamic(log_density, step_size)

    initial_states = vmap(mclmc.init, in_axes=(0, 0))(initial_parameters, keys)

    jit_step = jit(mclmc.step)

    states, infos = inference_loop_multiple_chains(
        keys, jit_step, initial_states, n_samples_per_chain
    )

    return states, infos
