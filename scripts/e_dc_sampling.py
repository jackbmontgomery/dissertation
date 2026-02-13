import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp
from jax._src.numpy.linalg import inv

from src.experiment import e_reaction_sampling_experiment
from src.sampling import (
    HMCSamplingAlgorithm,
    MalaSamplingAlgorithm,
    MCLMCSamplingAlgorithm,
    MetropolisHastingSamplingAlgorithm,
    NutsSamplingAlgorithm,
    PathfinderSamplingAlgorithm,
)
from src.voltammetry import LinearSweepDC


def main():
    voltammetry = LinearSweepDC()

    # --- Metropolis-Hasting ---
    # sampling_algorithm = MetropolisHastingSamplingAlgorithm(
    #     n_samples=40_000,
    #     sigma=jnp.array([0.01, 0.01, 0.01, 0.01]),
    # )
    # e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- MCLMC ---
    # sampling_algorithm = MCLMCSamplingAlgorithm(10_000, 1e-2)
    # e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- Pathfinder ---
    # sampling_algorithm = PathfinderSamplingAlgorithm(40_000, 1e-2)
    # e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- MALA ---
    # sampling_algorithm = MalaSamplingAlgorithm(n_samples=1000, step_size=1e-2)
    # e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- NUTS ---
    sampling_algorithm = NutsSamplingAlgorithm(
        10_000, 2.5e-2, inv_mass_matrix=jnp.array([0.1, 0.1, 0.1, 0.1])
    )
    e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- HMC ---
    # sampling_algorithm = HMCSamplingAlgorithm(
    #     2000,
    #     step_size=1e-2,
    #     inv_mass_matrix=jnp.array([0.2, 0.2, 0.2, 0.2]),
    #     num_integration_steps=10,
    # )
    # e_reaction_sampling_experiment(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
