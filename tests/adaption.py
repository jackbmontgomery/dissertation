import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from time import perf_counter

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.diagnostics import potential_scale_reduction
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.sampling import (
    NUM_CPUS,
    NutsSamplingAlgorithm,
)
from src.utils import generate_noisy_samples, init_nuts
from src.voltammetry import LinearSweepDC


def main():
    voltammetry = LinearSweepDC()

    # --- Sampling ---

    key = jr.key(0)
    generate_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = EMechanismFDMSolver(voltammetry)

    true_params = EMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(1.2),
    )

    base_current = fdm_solver.solve(true_params)

    samples = generate_noisy_samples(
        10,
        base_current,
        0.1,
        key=key,
    )

    def logdensity_fn(params: EMechanismFDMParams, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = EMechanismFDMParams(
        alpha=jnp.linspace(0.5, 0.7, NUM_CPUS),
        K0=jnp.linspace(5.0, 15.0, NUM_CPUS),
        E0=jnp.linspace(1.5, 2.5, NUM_CPUS),
        dB=jnp.linspace(0.8, 1.4, NUM_CPUS),
    )

    adapt_init_params = EMechanismFDMParams(
        alpha=jnp.array(0.5),
        K0=jnp.array(5.0),
        E0=jnp.array(1.5),
        dB=jnp.array(0.8),
    )

    print("--- Running Sampling ---")
    start_time = perf_counter()

    parameters = init_nuts(key, adapt_init_params, logdensity_fn)

    print(parameters)

    sampling_algorithm = NutsSamplingAlgorithm(80, **parameters)

    samples, logdensity, info = sampling_algorithm(
        sampling_key, init_params, logdensity_fn
    )

    samples.alpha.block_until_ready()

    end_time = perf_counter()
    print("--- Done ---")

    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Number of Samples: {len(samples.alpha.flatten())}")
    print(f"Data Type: {samples.alpha.dtype}")

    for k, v in info.items():
        print(f"{k}: {v}")


main()
