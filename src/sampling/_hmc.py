from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optax
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from jax import jit
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params

from ._base import AbstractSampler, LogDensity, inference_loop_multiple_chains


class HMCSampler(AbstractSampler):
    def __init__(self, log_density: LogDensity, n_samples: int, num_chains: int):
        self.log_density = log_density
        self.samples_per_chain = n_samples // num_chains

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
