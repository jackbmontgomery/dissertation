from time import perf_counter
from typing import Dict

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap

from src.fdm import AbstractFDSolver
from src.optimisers import make_adam_optimise, make_cmaes_optimise
from src.params import Params
from src.reaction._base import AbstractReaction
from src.utils import generate_noisy_samples, pretty_header


def optimisation_experiment(
    reaction: AbstractReaction,
    fd_solver: AbstractFDSolver,
    num_iterations: int,
    num_params: int,
    num_experimental_samples: int = 10,
    noise_percentage: float = 0.02,
    *,
    cmaes_params: Dict = {},
    adam_params: Dict = {},
    seed: int,
    save: bool,
):
    print(pretty_header(reaction))
    print(f"Noise Percentage: {noise_percentage:.2f}")

    key = jr.key(seed)
    key_samples, key_init = jr.split(key, 2)
    base_current = fd_solver.solve(reaction.true_parameters)

    experimental_samples = generate_noisy_samples(
        num_experimental_samples,
        base_current,
        noise_percentage,
        key=key_samples,
    )

    def logdensity_fn(params: Params):
        current = fd_solver.solve(params)
        return -jnp.sum((experimental_samples - current) ** 2)

    init_params = reaction.create_init_params(key_init, num_params)

    cmaes_optimise = make_cmaes_optimise(num_iterations, logdensity_fn, **cmaes_params)
    adam_optimise = make_adam_optimise(num_iterations, logdensity_fn, **adam_params)

    start_time = perf_counter()

    print(pretty_header("CMA-ES", char="-"))
    cmaes_keys = jr.split(key, num_params)
    _, cmaes_ld, cmaes_pp = vmap(cmaes_optimise)(init_params, cmaes_keys)
    cmaes_ld.block_until_ready()
    cmaes_done_time = perf_counter()
    print(f"Time Taken: {cmaes_done_time - start_time:.4f}")

    print(pretty_header("ADAM", char="-"))
    _, adam_ld, adam_pp = vmap(adam_optimise)(init_params)
    adam_ld.block_until_ready()
    adam_done_time = perf_counter()
    print(f"Time Taken: {adam_done_time - cmaes_done_time:.4f}")

    if save:
        mode_logdensity = logdensity_fn(reaction.true_parameters)
        file_name = f"reaction={reaction},noise={noise_percentage},seed={seed}"
        np.savez_compressed(
            f"./data/optimisation/{file_name}.npz",
            adam_ld=adam_ld,
            adam_pp=adam_pp,
            cmaes_ld=cmaes_ld,
            cmaes_pp=cmaes_pp,
            mode_logdensity=mode_logdensity,
        )
