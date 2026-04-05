from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params

from ._base import AbstractSampler, LogDensity, inference_loop_multiple_chains


class RWMHSampler(AbstractSampler):
    def __init__(self, log_density: LogDensity, n_samples: int, num_chains: int):
        self.log_density = log_density
        self.num_chains = num_chains
        self.num_samples_per_chain = n_samples // num_chains

    def run(
        self, init_params: Params, sampler_params: Dict, *, key: PRNGKeyArray
    ) -> Tuple[Params, Dict[str, Scalar]]:
        keys = jr.split(key, self.num_chains)

        rwmh = blackjax.additive_step_random_walk(self.log_density, **sampler_params)
        jit_step = jit(rwmh.step)

        init_states = vmap(rwmh.init, in_axes=(0, None))(init_params, self.log_density)

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, self.num_samples_per_chain
        )

        samples: Params = states.position

        info = {"Average Acceptance": jnp.round(jnp.mean(infos.is_accepted), 4)}

        return samples, info
