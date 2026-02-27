import multiprocessing
from typing import Callable, Dict, NamedTuple, Tuple

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit, pmap, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from src.utils import bfgs_minimise

NUM_CPUS = multiprocessing.cpu_count()

LogDensity = Callable[[PyTree], Array]


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
    ) -> Tuple[PyTree, Scalar, Dict]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AdditiveStepRandomWalkSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, sigma: Scalar):
        self.n_samples = n_samples
        self.sigma = sigma

    def __str__(self):
        return "MetropolisHastings"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Scalar, Dict]:
        print("--- Running Random Walk Metropolis-Hastings ---")

        keys = jr.split(key, NUM_CPUS)

        rw = blackjax.additive_step_random_walk(
            log_density, blackjax.mcmc.random_walk.normal(self.sigma)
        )

        jit_step = jit(rw.step)
        init_states = vmap(rw.init)(init_params)

        num_samples_per_chain = self.n_samples // NUM_CPUS

        print("--- Running Sampling ---")

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, num_samples_per_chain
        )

        avg_acceptance = jnp.mean(infos.is_accepted)

        sampling_info = {
            "Average Acceptance": f"{avg_acceptance:.2f}",
        }

        samples: PyTree = states.position
        logdensity: Scalar = states.logdensity

        print("--- Sampling Done ---")

        return samples, logdensity, sampling_info


class HMCSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(
        self,
        n_samples: int,
        learning_rate: float,
        initial_step_size: float,
        warmup_steps: int,
    ):
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.initial_step_size = initial_step_size
        self.warmup_steps = warmup_steps

    def __str__(self):
        return "HMC"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Scalar, Dict]:
        # Cheese Adaption / Warmup

        print("--- Running HMC ---")

        warmup_key, key = jr.split(key, 2)

        warmup = blackjax.chees_adaptation(log_density, NUM_CPUS)

        key_warmup, key_sample = jr.split(warmup_key)

        optim = optax.adam(self.learning_rate)

        print("--- Running Chees Warmup ---")

        (initial_states, parameters), _ = warmup.run(
            key_warmup,
            init_params,
            step_size=self.initial_step_size,
            optim=optim,
            num_steps=self.warmup_steps,
            max_sampling_steps=1000,
        )

        jax.debug.print("Parameters: {x}", x=parameters)

        hmc = blackjax.dynamic_hmc(log_density, **parameters)

        jit_step = jit(hmc.step)

        keys = jr.split(key, NUM_CPUS)

        samples_per_chain = self.n_samples // NUM_CPUS

        print("--- Running Sampling ---")

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, initial_states, samples_per_chain
        )

        algo_info = {
            "Average Acceptance": f"{jnp.mean(infos.acceptance_rate):.2f}",
            "Average Integration Steps": f"{jnp.mean(infos.num_integration_steps):.2f}",
        }

        samples: PyTree = states.position
        logdensity: Scalar = states.logdensity

        print("--- Done ---")

        return samples, logdensity, algo_info


class MCHMCSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(self, n_samples: int, step_size: float):
        self.n_samples = n_samples
        self.step_size = step_size

    def __str__(self):
        return "MCHMC"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Scalar, Dict]:
        print("--- Running MCHMC ---")
        init_key, inference_key = jr.split(key)

        mchmc = blackjax.adjusted_mclmc_dynamic(log_density, self.step_size)

        # init_params = vmap(bfgs_minimise, in_axes=(0, None))(init_params, log_density)

        init_keys = jr.split(init_key, NUM_CPUS)
        mchmc_kernel = jit(mchmc.step)
        initial_states = vmap(mchmc.init)(init_params, init_keys)

        inference_keys = jr.split(inference_key, NUM_CPUS)
        samples_per_chain = self.n_samples // NUM_CPUS
        states, infos = inference_loop_multiple_chains(
            inference_keys, mchmc_kernel, initial_states, samples_per_chain
        )

        algo_info = {
            "Average Acceptance": f"{jnp.mean(infos.acceptance_rate):.2f}",
            "Average Integration Steps": f"{jnp.mean(infos.num_integration_steps):.2f}",
        }

        samples: PyTree = states.position
        logdensity: Scalar = states.logdensity

        return samples, logdensity, algo_info
