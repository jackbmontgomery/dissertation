import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.diagnostics import potential_scale_reduction

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.sampling import (
    MetropolisHastingsSamplingAlgorithm,
)
from src.utils import generate_noisy_samples
from src.voltammetry import LinearSweepDC


def main():
    voltammetry = LinearSweepDC()

    # --- Metropolis-Hasting ---

    sampling_algorithm = MetropolisHastingsSamplingAlgorithm(
        n_samples=40_000,
        sigma=jnp.array([0.01, 0.01, 0.01, 0.01]),
    )

    # --- Sampling ---
    key = jr.key(42)
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

    key, k1, k2, k3, k4 = jr.split(key, 5)

    init_params = EMechanismFDMParams(
        alpha=jax.random.uniform(k1, shape=(8,), minval=0.0, maxval=1.0),
        K0=jax.random.uniform(k2, shape=(8,), minval=1.0, maxval=50.0),
        E0=jax.random.uniform(k3, shape=(8,), minval=0.0, maxval=5.0),
        dB=jax.random.uniform(k4, shape=(8,), minval=0.0, maxval=1.0),
    )

    print("--- Running Sampling ---")
    start_time = perf_counter()
    samples, info = sampling_algorithm(sampling_key, init_params, log_density)
    samples.alpha.block_until_ready()
    end_time = perf_counter()
    print("--- Done ---")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    for k, v in info.items():
        print(f"{k}: {v}")

    samples = jnp.squeeze(jnp.array([jax.tree.leaves(samples)]), 0)
    x = jax.vmap(potential_scale_reduction)(samples)
    print(x)


if __name__ == "__main__":
    main()
