from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optax
from blackjax.mcmc.hmc import HMCState
from jax import jit
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params

from ._base import AbstractSampler, LogDensity, inference_loop_multiple_chains


class NUTSSampler(AbstractSampler):
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
    ) -> Tuple[HMCState, Dict]:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            self.log_density,
            initial_step_size=initial_step_size,
        )

        (last_state, nuts_params), warmup_info = warmup.run(
            key,
            params,
            num_steps=steps,
        )

        return last_state, nuts_params

    def run(
        self,
        init_states: HMCState,
        n_samples: int,
        *,
        key: PRNGKeyArray,
        nuts_params: Dict,
    ) -> Tuple[Params, Dict[str, Scalar]]:
        nuts = blackjax.nuts(self.log_density, **nuts_params)

        jit_step = jit(nuts.step)

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
