import multiprocessing
from typing import Callable, Dict, NamedTuple, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from jax import jit, pmap, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, PyTree

from src.params import ElectrodeKineticsParameters

NUM_CPUS = multiprocessing.cpu_count()

LogDensity = Callable[[ElectrodeKineticsParameters], Array]

SamplingFunction = Callable[
    [PRNGKeyArray, ElectrodeKineticsParameters, LogDensity],
    Tuple[ElectrodeKineticsParameters, Dict],
]


def _inference_loop(
    key: PRNGKeyArray,
    kernel: Callable[
        [PRNGKeyArray, ElectrodeKineticsParameters], Tuple[PyTree, PyTree]
    ],
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


def _bfgs_minimise(initial_parameters, log_density: LogDensity):
    bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
    solution = optx.minimise(
        lambda params, _: -log_density(params), bfgs, initial_parameters
    )
    sampling_init_params = solution.value
    return sampling_init_params


def metropolis_hastings_sampling(
    key: PRNGKeyArray,
    initial_parameters: ElectrodeKineticsParameters,
    log_density: LogDensity,
    *,
    n_samples: int = 40_000,
    sigma: Array = jnp.array([0.01, 0.01, 0.01]),
) -> Tuple[ElectrodeKineticsParameters, Dict]:
    rw = blackjax.additive_step_random_walk(
        log_density, blackjax.mcmc.random_walk.normal(sigma)
    )

    sampling_init_params = _bfgs_minimise(initial_parameters, log_density)

    initial_state = rw.init(sampling_init_params, key)

    jit_step = jit(rw.step)

    states, infos = _inference_loop(key, jit_step, initial_state, n_samples)

    avg_acceptance = jnp.mean(infos.is_accepted)

    sampling_info = {"Average Acceptance": f"{avg_acceptance:.2f}"}

    samples: ElectrodeKineticsParameters = states.position

    return samples, sampling_info


def mclmc_sampling(
    key: PRNGKeyArray,
    initial_parameters: ElectrodeKineticsParameters,
    log_density: LogDensity,
    *,
    n_samples: int = 8_000,
    step_size: float = 1e-2,
) -> Tuple[ElectrodeKineticsParameters, Dict]:
    mclmc = blackjax.adjusted_mclmc_dynamic(log_density, step_size)

    sampling_init_params = _bfgs_minimise(initial_parameters, log_density)

    sampling_init_params = ElectrodeKineticsParameters(
        alpha=jnp.full((NUM_CPUS,), initial_parameters.alpha),
        kappa=jnp.full((NUM_CPUS,), initial_parameters.kappa),
        epsilon=jnp.full((NUM_CPUS,), initial_parameters.epsilon),
    )

    keys = jr.split(key, NUM_CPUS)

    initial_states = vmap(mclmc.init, in_axes=(0, 0))(sampling_init_params, keys)

    n_samples_per_chain = n_samples // NUM_CPUS

    jit_step = jit(mclmc.step)

    states, infos = inference_loop_multiple_chains(
        keys, jit_step, initial_states, n_samples_per_chain
    )

    avg_acceptance = jnp.mean(infos.is_accepted)

    sampling_info = {"Average Acceptance": f"{avg_acceptance:.2f}"}

    samples: ElectrodeKineticsParameters = states.position

    return samples, sampling_info


def pathfinder_sampling(
    key: PRNGKeyArray,
    initial_parameters: ElectrodeKineticsParameters,
    log_density: LogDensity,
    *,
    n_samples: int = 40_000,
    step_size: float = 1e-2,
) -> Tuple[ElectrodeKineticsParameters, Dict]:
    approx_key, sample_key = jr.split(key)
    pathfinder = blackjax.pathfinder(log_density)

    print("--- Approximating ---")
    state, info = pathfinder.approximate(approx_key, initial_parameters, ftol=1e-8)

    info = {"Elbo Path": info.path.elbo}

    print("--- Sampling ---")
    samples, _ = pathfinder.sample(sample_key, state, n_samples)

    return samples, info
