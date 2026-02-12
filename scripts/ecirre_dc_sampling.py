import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import (
    e_reaction_sampling_experiment,
    ecirre_reaction_sampling_experiment,
)
from src.sampling import (
    MalaSamplingAlgorithm,
    MCLMCSamplingAlgorithm,
    MetropolisHastingSamplingAlgorithm,
    PathfinderSamplingAlgorithm,
)
from src.voltammetry import LinearSweepDC


def main():
    voltammetry = LinearSweepDC()

    # --- Metropolis-Hasting ---
    # sampling_algorithm = MetropolisHastingSamplingAlgorithm(
    #     n_samples=1000,
    #     sigma=jnp.full((6,), 0.01),
    # )
    # ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- MCLMC ---
    # sampling_algorithm = MCLMCSamplingAlgorithm(5_000, 1e-2)
    # ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- Pathfinder ---
    sampling_algorithm = PathfinderSamplingAlgorithm(40_000, 1e-2)
    ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- MALA ---
    # sampling_algorithm = MalaSamplingAlgorithm(n_samples=1000, step_size=1e-2)
    # ecirre_reaction_sampling_experiment(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
