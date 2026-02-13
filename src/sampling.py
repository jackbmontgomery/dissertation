import multiprocessing
from typing import Callable, Dict, NamedTuple, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jax import jit, pmap, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from src.utils import bfgs_minimise

NUM_CPUS = multiprocessing.cpu_count()

LogDensity = Callable[[PyTree], Array]


def repeat_params(params: PyTree, n: int):
    return jt.map(lambda x: jnp.full((n,), x), params)


def flatten_state_positions(state_positions: PyTree):
    return jt.map(lambda x: x.flatten(), state_positions)


def inference_loop(
    key: PRNGKeyArray,
    kernel: Callable[[PRNGKeyArray, PyTree], Tuple[PyTree, PyTree]],
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
    inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
)


class AbstractSamplingAlgorithm:
    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MetropolisHastingSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, sigma: Scalar):
        self.n_samples = n_samples
        self.sigma = sigma

    def __str__(self):
        return "MetropolisHasting"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        rw = blackjax.additive_step_random_walk(
            log_density, blackjax.mcmc.random_walk.normal(self.sigma)
        )

        sampling_init_params = bfgs_minimise(init_params, log_density)

        initial_state = rw.init(sampling_init_params, key)

        jit_step = jit(rw.step)

        states, infos = inference_loop(key, jit_step, initial_state, self.n_samples)

        avg_acceptance = jnp.mean(infos.is_accepted)

        sampling_info = {"Average Acceptance": f"{avg_acceptance:.2f}"}

        samples: PyTree = states.position

        return samples, sampling_info


class MalaSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, step_size: float):
        self.n_samples = n_samples
        self.step_size = step_size

    def __str__(self):
        return "Mala"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        mala = blackjax.mala(log_density, self.step_size)

        init_params = bfgs_minimise(init_params, log_density)

        initial_state = mala.init(init_params)
        jit_step = jit(mala.step)

        states, infos = inference_loop(key, jit_step, initial_state, self.n_samples)

        avg_acceptance = jnp.mean(infos.is_accepted)

        sampling_info = {"Average Acceptance": f"{avg_acceptance:.2f}"}

        samples: PyTree = states.position

        return samples, sampling_info


class MCLMCSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, step_size: float):
        self.n_samples = n_samples
        self.step_size = step_size

    def __str__(self):
        return "MCLMC"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        mclmc = blackjax.adjusted_mclmc_dynamic(log_density, self.step_size)

        sampling_init_params = bfgs_minimise(init_params, log_density)

        sampling_init_params = repeat_params(sampling_init_params, NUM_CPUS)

        keys = jr.split(key, NUM_CPUS)

        initial_states = vmap(mclmc.init, in_axes=(0, 0))(sampling_init_params, keys)

        n_samples_per_chain = self.n_samples // NUM_CPUS

        jit_step = jit(mclmc.step)

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, initial_states, n_samples_per_chain
        )

        avg_acceptance = jnp.mean(infos.is_accepted)

        sampling_info = {"Average Acceptance": f"{avg_acceptance:.2f}"}

        samples = flatten_state_positions(states.position)

        return samples, sampling_info


class NutsSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, step_size: float, inv_mass_matrix: Scalar):
        self.n_samples = n_samples
        self.step_size = step_size
        self.inv_mass_matrix = inv_mass_matrix

    def __str__(self):
        return "Nuts"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        nuts = blackjax.nuts(log_density, self.step_size, self.inv_mass_matrix)

        jit_step = jit(nuts.step)

        init_params = bfgs_minimise(init_params, log_density)

        init_params = repeat_params(init_params, NUM_CPUS)
        keys = jr.split(key, NUM_CPUS)
        samples_per_chain = self.n_samples // NUM_CPUS

        initial_states = vmap(nuts.init)(init_params)

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, initial_states, samples_per_chain
        )

        algo_info = {
            "Average Acceptance": f"{jnp.mean(infos.acceptance_rate):.4f}",
            "Average Integration Steps": jnp.mean(infos.num_integration_steps),
        }

        samples = flatten_state_positions(states.position)

        return samples, algo_info


class HMCSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(
        self,
        n_samples: int,
        step_size: float,
        inv_mass_matrix: Scalar,
        num_integration_steps: int,
    ):
        self.n_samples = n_samples
        self.step_size = step_size
        self.inv_mass_matrix = inv_mass_matrix
        self.num_integration_steps = num_integration_steps

    def __str__(self):
        return "HMC"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        hmc = blackjax.hmc(
            log_density,
            self.step_size,
            self.inv_mass_matrix,
            self.num_integration_steps,
        )

        init_params = bfgs_minimise(init_params, log_density)
        init_state = hmc.init(init_params)
        jit_step = jit(hmc.step)

        states, infos = inference_loop(key, jit_step, init_state, self.n_samples)

        algo_info = {
            "Average Acceptance": f"{jnp.mean(infos.is_accepted):.4f}",
            "Average Integration Steps": jnp.mean(infos.num_integration_steps),
        }

        samples = flatten_state_positions(states.position)

        return samples, algo_info


class PathfinderSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, step_size: float):
        self.n_samples = n_samples
        self.step_size = step_size

    def __str__(self):
        return "Pathfinder"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Dict]:
        approx_key, sample_key = jr.split(key)
        pathfinder = blackjax.pathfinder(log_density)

        print("--- Approximating ---")
        state, info = pathfinder.approximate(
            approx_key, init_params, gtol=1e-5, ftol=1e-5
        )

        info = {"Elbo Path": info.path.elbo}

        print("--- Sampling ---")
        samples, _ = pathfinder.sample(sample_key, state, self.n_samples)

        return samples, info
