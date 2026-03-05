import multiprocessing
from time import perf_counter
from typing import Callable, NamedTuple, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit, pmap, vmap
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from src.utils import adam_minimise

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
    ) -> Tuple[PyTree, Scalar]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AdditiveStepRandomWalkSamplingAlgorithm(AbstractSamplingAlgorithm):
    def __init__(
        self,
        n_samples: int,
        sigma: Scalar,
        burn_in_lr: float = 1e-1,
        burn_in_steps: int = 200,
    ):
        self.n_samples = n_samples
        self.sigma = sigma
        self.burn_in_lr = burn_in_lr
        self.burn_in_steps = burn_in_steps

    def __str__(self):
        return "RW"

    def __call__(
        self, key: PRNGKeyArray, init_params: PyTree, log_density: LogDensity
    ) -> Tuple[PyTree, Scalar]:
        start_time = perf_counter()

        print("--- Running Random Walk Metropolis-Hastings ---")

        print("Adam Minimise - ", end="")
        init_params, log_likelihoods = vmap(
            adam_minimise, in_axes=(0, None, None, None)
        )(init_params, self.burn_in_lr, self.burn_in_steps, log_density)
        for i, log_likelihood in enumerate(log_likelihoods):
            print(
                f"Chain {i + 1}: Start: {log_likelihood[0]:.4f} - End: {log_likelihood[-2]:.4f}"
            )

        warm_up_time = perf_counter()
        print(f"Time Taken: {warm_up_time - start_time:.4f}")
        keys = jr.split(key, NUM_CPUS)

        rw = blackjax.additive_step_random_walk(
            log_density, blackjax.mcmc.random_walk.normal(self.sigma)
        )

        jit_step = jit(rw.step)
        init_states = vmap(rw.init)(init_params)

        num_samples_per_chain = self.n_samples // NUM_CPUS

        print("Running Sampling - ", end="")

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, num_samples_per_chain
        )

        avg_acceptance = jnp.mean(infos.is_accepted)

        samples: PyTree = states.position
        logdensity: Scalar = states.logdensity

        print("--- Sampling Done ---")
        sampling_time = perf_counter()
        print(f"Time Taken: {sampling_time - warm_up_time:.4f}")
        print(f"Average Acceptance: {avg_acceptance:.2f}")

        return samples, logdensity


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
    ) -> Tuple[PyTree, Scalar]:
        start_time = perf_counter()

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
        )

        print("Step size:", f"{parameters['step_size']:.4f}")
        print("Inverse Mass Matrix", parameters["inverse_mass_matrix"])
        warm_up_time = perf_counter()

        print(f"Chees Warmup Time: {warm_up_time - start_time:.4f}")

        hmc = blackjax.dynamic_hmc(log_density, **parameters)

        jit_step = jit(hmc.step)

        keys = jr.split(key, NUM_CPUS)

        samples_per_chain = self.n_samples // NUM_CPUS

        print("--- Running Sampling ---")

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, initial_states, samples_per_chain
        )

        samples: PyTree = states.position
        logdensity: Scalar = states.logdensity

        print("--- Sampling Done ---")
        sampling_time = perf_counter()
        print(f"Sampling Time: {sampling_time - warm_up_time:.4f}")

        print(f"Average Acceptance: {jnp.mean(infos.acceptance_rate):.2f}")
        print(f"Average Integration Steps: {jnp.mean(infos.num_integration_steps):.2f}")

        return samples, logdensity
