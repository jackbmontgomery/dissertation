import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import HeterogeneousReactionSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    HMCSamplingAlgorithm,
)


def main():
    sampling_experiment = HeterogeneousReactionSamplingExperiment()
    sigma = 10
    noise = 0.1

    # --- Metropolis-Hasting ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=8000,
        sigma=jnp.repeat(0.001, 10),
    )
    sampling_experiment.run(sampling_algorithm, noise, sigma)

    # --- HMC ---
    # sampling_algorithm = HMCSamplingAlgorithm(4_000, 1e-2, 1e-3, 100)
    # sampling_experiment.run(sampling_algorithm, noise, sigma)


if __name__ == "__main__":
    main()
