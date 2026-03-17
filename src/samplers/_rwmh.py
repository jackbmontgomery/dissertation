from typing import Dict, Tuple

import blackjax
import jax.numpy as jnp
import jax.random as jr
from blackjax.mcmc.random_walk import RWState, init
from jax import jit, vmap
from jaxtyping import PRNGKeyArray, Scalar

from src.optimisers import make_adam_optimise
from src.params import Params
from src.samplers._base import AbstractSampler

from ._base import LogDensity, inference_loop_multiple_chains


class RWMHSampler(AbstractSampler):
    def __init__(self, log_density: LogDensity, num_chains: int):
        self.log_density = log_density
        self.num_chains = num_chains

    def warmup(
        self, params: Params, learning_rate: float, steps: int
    ) -> Tuple[RWState, Dict[str, Scalar]]:
        adam_minimise = make_adam_optimise(
            num_steps=steps, log_density=self.log_density, learning_rate=learning_rate
        )
        warmed_up_params, _, _ = vmap(adam_minimise)(params)
        init_state: RWState = vmap(init, in_axes=(0, None))(
            warmed_up_params, self.log_density
        )
        return init_state, {}

    def run(
        self, init_states: RWState, n_samples: int, *, key: PRNGKeyArray, sigma: Scalar
    ) -> Tuple[Params, Dict[str, Scalar]]:
        rw = blackjax.additive_step_random_walk(
            self.log_density, blackjax.mcmc.random_walk.normal(sigma)
        )

        jit_step = jit(rw.step)

        keys = jr.split(key, self.num_chains)
        num_samples_per_chain = n_samples // self.num_chains

        states, infos = inference_loop_multiple_chains(
            keys, jit_step, init_states, num_samples_per_chain
        )

        samples: Params = states.position

        info = {"avg_acceptance": jnp.round(jnp.mean(infos.is_accepted), 4)}

        return samples, info
