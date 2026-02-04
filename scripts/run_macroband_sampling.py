import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import MacroElectrodeFDMSolver
from src.params import ElectrodeKineticsParameters
from src.sampling import (
    SamplingFunction,
    mclmc_sampling,
    metropolis_hastings_sampling,
    pathfinder_sampling,
)
from src.utils import generate_noisy_samples
from src.voltammetry import AbstractVoltammetryTechnique, LinearSweepAC, LinearSweepDC


@dataclass
class Args:
    v: Literal["ac", "dc"]
    s: Literal["mh", "mclmc", "pathfinder"]
    seed: int = 42


def main(
    voltammetry: AbstractVoltammetryTechnique,
    sampling_algo: SamplingFunction,
    data_file: str,
    seed: int,
):
    print(f"--- Running {data_file} ---")

    key = jr.key(seed)
    generate_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = MacroElectrodeFDMSolver(voltammetry, 1e-2, 5e-2)

    experimental_params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.6), kappa=jnp.array(100.0), epsilon=jnp.array(2.0)
    )

    samples = generate_noisy_samples(
        10,
        experimental_params,
        0.1,
        fdm_solver,
        key=key,
    )

    def log_density(params: ElectrodeKineticsParameters, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.3), kappa=jnp.array(80.0), epsilon=jnp.array(0.0)
    )

    start_time = perf_counter()
    samples, info = sampling_algo(sampling_key, init_params, log_density)
    np.savez_compressed(
        f"./data/{data_file}",
        alpha=samples.alpha.flatten(),
        kappa=samples.kappa.flatten(),
        epsilon=samples.epsilon.flatten(),
    )
    end_time = perf_counter()
    print("--- Done ---")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Number of Samples: {len(samples.alpha.flatten())}")
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)

    if args.v == "ac":
        voltammetry = LinearSweepAC()
    elif args.v == "dc":
        voltammetry = LinearSweepDC()
    else:
        raise Exception("Invalid voltammetry selection")

    if args.s == "mh":
        sampling_algo: SamplingFunction = metropolis_hastings_sampling

    elif args.s == "mclmc":
        sampling_algo: SamplingFunction = mclmc_sampling

    elif args.s == "pathfinder":
        sampling_algo: SamplingFunction = pathfinder_sampling

    else:
        raise Exception("Invalid sampling selection")

    data_file = f"{args.v}_{args.s}.npz"

    main(voltammetry, sampling_algo, data_file, args.seed)
