from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import ECirreMechanismFDMSolver, EMechanismFDMSolver
from src.params import ECirreMechanismFDMParams, EMechanismFDMParams
from src.sampling import AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import AbstractVoltammetryTechnique

"""
Defines a sampling experiment which would be:
1. A voltammetry technique
2. A sampling algorithm
3. A seed for reproducibiliy
"""


def e_reaction_sampling_experiment(
    sampling_algorithm: AbstractSamplingAlgorithm,
    voltammetry: AbstractVoltammetryTechnique,
    seed: int = 0,
):
    print(f"--- E Setting up {sampling_algorithm} {voltammetry} ---")

    key = jr.key(seed)
    generate_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = EMechanismFDMSolver(voltammetry)

    true_params = EMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(0.5),
    )

    base_current = fdm_solver.solve(true_params)

    samples = generate_noisy_samples(
        10,
        base_current,
        0.1,
        key=key,
    )

    def log_density(params: EMechanismFDMParams, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = EMechanismFDMParams(
        alpha=jnp.array(0.3),
        K0=jnp.array(20.0),
        E0=jnp.array(0.0),
        dB=jnp.array(0.8),
    )

    print("--- Running Sampling ---")
    start_time = perf_counter()
    samples, info = sampling_algorithm(sampling_key, init_params, log_density)
    samples.alpha.block_until_ready()
    end_time = perf_counter()
    print("--- Done ---")

    data_file = f"E_{sampling_algorithm}_{voltammetry}.npz"
    np.savez_compressed(
        f"./data/{data_file}",
        alpha=samples.alpha.flatten(),
        K0=samples.K0.flatten(),
        E0=samples.E0.flatten(),
        dB=samples.dB.flatten(),
    )
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Number of Samples: {len(samples.alpha.flatten())}")
    print(f"Data Type: {samples.alpha.dtype}")
    for k, v in info.items():
        print(f"{k}: {v}")


def ecirre_reaction_sampling_experiment(
    sampling_algorithm: AbstractSamplingAlgorithm,
    voltammetry: AbstractVoltammetryTechnique,
    seed: int = 0,
):
    print(f"--- Setting up ECirre {sampling_algorithm} {voltammetry} ---")

    key = jr.key(seed)
    generate_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = ECirreMechanismFDMSolver(voltammetry)

    true_params = ECirreMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        Kplus=jnp.array(10.0),
        Kminus=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(0.5),
    )

    base_current = fdm_solver.solve(true_params)

    samples = generate_noisy_samples(
        10,
        base_current,
        0.1,
        key=key,
    )

    def log_density(params: ECirreMechanismFDMParams, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = ECirreMechanismFDMParams(
        alpha=jnp.array(0.3),
        K0=jnp.array(20.0),
        Kplus=jnp.array(20.0),
        Kminus=jnp.array(5.0),
        E0=jnp.array(0.0),
        dB=jnp.array(0.8),
    )

    print("--- Running Sampling ---")
    start_time = perf_counter()
    samples, info = sampling_algorithm(sampling_key, init_params, log_density)
    samples.alpha.block_until_ready()
    end_time = perf_counter()
    print("--- Done ---")

    data_file = f"ECirre_{sampling_algorithm}_{voltammetry}.npz"
    np.savez_compressed(
        f"./data/{data_file}",
        alpha=samples.alpha,
        K0=samples.K0,
        Kplus=samples.Kplus,
        Kminus=samples.Kminus,
        E0=samples.E0,
        dB=samples.dB,
    )
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Number of Samples: {len(samples.alpha)}")
    print(f"Data Type: {samples.alpha.dtype}")
    for k, v in info.items():
        print(f"{k}: {v}")
