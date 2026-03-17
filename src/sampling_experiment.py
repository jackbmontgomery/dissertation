import multiprocessing
import os

from jax import flatten_util

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import gzip
import pickle
from time import perf_counter
from typing import Dict

import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from src.fdm import AbstractFDSolver, ElectronReactionFDSolver
from src.params import Params
from src.reaction import AbstractReaction, ElectronReaction
from src.samplers import HMCSampler, RWMHSampler
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC


def print_infos(info: Dict):
    for key, item in info.items():
        print(key, item)


def sampling_experiment(
    reaction: AbstractReaction,
    fd_solver: AbstractFDSolver,
    num_chains: int,
    num_experimental_samples: int = 10,
    experimental_noise: float = 0.25,
    *,
    num_rwmh_samples: int,
    rwmh_kwargs: Dict,
    num_hmc_samples: int,
    hmc_kwargs: Dict,
    seed: int,
    warmup_learning_rate: float = 1e-2,
    warmup_steps: int = 250,
):
    print(f"============ {reaction} ============")
    key = jr.key(seed)
    key_samples, key_init = jr.split(key, 2)
    _, base_current = fd_solver.solve(reaction.true_parameters)

    experimental_samples = generate_noisy_samples(
        num_experimental_samples,
        base_current,
        experimental_noise,
        key=key_samples,
    )

    def logdensity_fn(params: Params):
        _, current = fd_solver.solve(params)
        return -jnp.sum((experimental_samples - current) ** 2)

    init_params = reaction.create_init_params(key_init, num_chains)

    print("------------ Running RWMH ------------")
    start_time = perf_counter()

    key, key_rwmh = jr.split(key)

    rwmh = RWMHSampler(logdensity_fn, num_chains)

    init_states, _ = rwmh.warmup(
        init_params, learning_rate=warmup_learning_rate, steps=warmup_steps
    )

    flat_init_states, _ = flatten_util.ravel_pytree(init_states)
    flat_init_states.block_until_ready()

    warm_up_time = perf_counter()
    print(f"Warm Up Time: {warm_up_time - start_time:.4f}")
    rwmh_samples, infos = rwmh.run(
        init_states, num_rwmh_samples, key=key_rwmh, **rwmh_kwargs
    )

    print(f"Sampling Time: {perf_counter() - warm_up_time:.4f}")
    print_infos(infos)
    print("------------------------")

    print("------------ Running  HMC ------------")
    start_time = perf_counter()

    key, key_hmc = jr.split(key)
    hmc = HMCSampler(logdensity_fn, num_chains)

    key_warmup, key_hmc = jr.split(key, 2)
    init_states, hmc_params = hmc.warmup(
        init_params,
        learning_rate=warmup_learning_rate,
        steps=warmup_steps,
        key=key_warmup,
        **hmc_kwargs,
    )

    flat_init_states, _ = flatten_util.ravel_pytree(init_states)
    flat_init_states.block_until_ready()

    warm_up_time = perf_counter()
    print(f"Warm Up Time: {warm_up_time - start_time:.4f}")

    hmc_samples, infos = hmc.run(
        init_states, num_hmc_samples, key=key_hmc, hmc_params=hmc_params
    )

    print(f"Sampling Time: {perf_counter() - warm_up_time:.4f}")
    print_infos(infos)
    print("------------------------")

    file_name = f"reaction={reaction},noise={experimental_noise},seed={seed}"

    flat_hmc_samples, unravel_hmc = ravel_pytree(hmc_samples)
    flat_rwmh_samples, unravel_rwmh = ravel_pytree(rwmh_samples)

    with gzip.open(f"./data/sampling/{file_name}.pkl.gz", "wb") as f:
        pickle.dump({"hmc": hmc_samples, "rwmh": rwmh_samples}, f)

    print("================================================")
