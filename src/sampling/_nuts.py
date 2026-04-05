from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
import optax
from blackjax.mcmc.hmc import HMCState
from jax import jit, vmap
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params

from ._base import AbstractSampler, LogDensity, inference_loop_multiple_chains


class NUTSSampler(AbstractSampler):
    def __init__(self, log_density: LogDensity, n_samples: int, num_chains: int):
        self.log_density = log_density
        self.num_chains = num_chains
        self.samples_per_chain = n_samples // num_chains

    def run(
        self, init_params: Params, sampler_params: Dict, *, key: PRNGKeyArray
    ) -> Tuple[Params, Dict[str, Scalar]]:
        keys = jr.split(key, self.num_chains)

        nuts = blackjax.nuts(self.log_density, **sampler_params)
        jit_step = jit(nuts.step)

        init_states = vmap(nuts.init)(init_params)

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, self.samples_per_chain
        )
        samples: Params = states.position

        info = {
            "Average Integration Steps": jnp.round(
                jnp.mean(infos.num_integration_steps), 2
            ),
        }

        return samples, info
