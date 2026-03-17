from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optax
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from jax import jit
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params
from src.samplers._base import AbstractSampler

from ._base import LogDensity, inference_loop_multiple_chains


class HMCSampler(AbstractSampler):
    def __init__(self, log_density: LogDensity, num_chains: int):
        self.log_density = log_density
        self.num_chains = num_chains

    def warmup(
        self,
        params: Params,
        learning_rate: float,
        steps: int,
        *,
        key: PRNGKeyArray,
        initial_step_size: float,
    ) -> Tuple[DynamicHMCState, Dict]:
        warmup = blackjax.chees_adaptation(self.log_density, self.num_chains)

        optim = optax.adam(learning_rate)

        (initial_states, hmc_params), _ = warmup.run(
            key,
            params,
            step_size=initial_step_size,
            optim=optim,
            num_steps=steps,
        )

        return initial_states, hmc_params

    def run(
        self,
        init_states: DynamicHMCState,
        n_samples: int,
        *,
        key: PRNGKeyArray,
        hmc_params: Dict,
    ) -> Tuple[Params, Dict[str, Scalar]]:
        hmc = blackjax.dynamic_hmc(self.log_density, **hmc_params)

        jit_step = jit(hmc.step)

        keys = jr.split(key, self.num_chains)

        samples_per_chain = n_samples // self.num_chains

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, samples_per_chain
        )
        samples: Params = states.position

        info = {
            "avg_acceptance": jnp.round(jnp.mean(infos.is_accepted), 4),
            "avg_integration_step": jnp.round(jnp.mean(infos.num_integration_steps), 2),
        }

        return samples, info
