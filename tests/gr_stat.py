import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.sampling import (
    MetropolisHastingsSamplingAlgorithm,
)
from src.utils import bfgs_minimise, generate_noisy_samples
from src.voltammetry import CyclicDC, LinearSweepAC, LinearSweepDC


def main():
    voltammetry = LinearSweepAC()

    # --- Metropolis-Hasting ---

    sampling_algorithm = MetropolisHastingsSamplingAlgorithm(
        n_samples=160_000,
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
        dB=jnp.array(1.2),
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

    def linspace_init(low, high, n_chains=8):
        return jnp.linspace(low, high, n_chains)

    init_params = EMechanismFDMParams(
        alpha=linspace_init(0.5, 0.7),
        K0=linspace_init(5.0, 15.0),
        E0=linspace_init(1.5, 2.5),
        dB=linspace_init(0.8, 1.4),
    )

    # init_params = jax.vmap(bfgs_minimise, in_axes=(0, None))(init_params, log_density)

    print("--- Running Sampling ---")
    start_time = perf_counter()
    samples, info = sampling_algorithm(sampling_key, init_params, log_density)
    samples.alpha.block_until_ready()
    end_time = perf_counter()
    print("--- Done ---")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print("Average Acceptance:", info["Average Acceptance"])

    # for k, v in info.items():
    #     print(f"{k}: {v}")

    # vmap(potential_scale_reduction)()

    # def flatten_state_positions(state_positions):
    #     return jt.map(lambda x: x.flatten(), state_positions)
    #

    np.savez_compressed(
        "./data/temp_ac.npz",
        alpha=samples.alpha.squeeze(),
        K0=samples.K0.squeeze(),
        E0=samples.E0.squeeze(),
        dB=samples.dB.squeeze(),
        logdensity=info["logdensity"],
    )


if __name__ == "__main__":
    main()
