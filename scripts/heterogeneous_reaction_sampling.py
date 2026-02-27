import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import HeterogeneousReactionSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    MCHMCSamplingAlgorithm,
)
from src.voltammetry import CyclicDC


def main():
    voltammetry = CyclicDC()
    sampling_experiment = HeterogeneousReactionSamplingExperiment()

    # --- Metropolis-Hasting ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=80_000,
        sigma=jnp.repeat(0.001, 10),
    )
    sampling_experiment.run(sampling_algorithm, voltammetry)

    # --- MCHMC ---
    # sampling_algorithm = MCHMCSamplingAlgorithm(n_samples=8000, step_size=5e-3)
    # sampling_experiment.run(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
