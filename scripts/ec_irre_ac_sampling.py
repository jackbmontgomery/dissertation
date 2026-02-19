import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import ecirre_reaction_sampling_experiment
from src.sampling import (
    MetropolisHastingsSamplingAlgorithm,
    NutsSamplingAlgorithm,
    PathfinderSamplingAlgorithm,
)
from src.voltammetry import LinearSweepAC


def main():
    voltammetry = LinearSweepAC()

    # --- Metropolis-Hasting ---
    sampling_algorithm = MetropolisHastingsSamplingAlgorithm(
        n_samples=40_000,
        sigma=jnp.full((6,), 0.01),
    )
    ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- Pathfinder ---
    sampling_algorithm = PathfinderSamplingAlgorithm(40_000, 1e-2)
    ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- Nuts ---
    sampling_algorithm = NutsSamplingAlgorithm(
        10_000, 4e-2, inverse_mass_matrix=jnp.repeat(0.05, 6)
    )
    ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
