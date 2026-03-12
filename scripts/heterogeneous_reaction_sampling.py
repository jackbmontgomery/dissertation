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

hyperparams = {
    "RW": {
        10: jnp.repeat(0.075, 7),
        100: jnp.repeat(0.025, 7),
        1000: jnp.repeat(0.01, 7),
    }
}


def main():
    sampling_experiment = HeterogeneousReactionSamplingExperiment()
    sigma = 1000
    noise = 0.25

    # --- Metropolis-Hasting ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=80_000,
        sigma=hyperparams["RW"][sigma],
    )
    sampling_experiment.run(sampling_algorithm, noise, sigma, "DC")

    # --- HMC ---
    sampling_algorithm = HMCSamplingAlgorithm(16_000, 1e-2, 1e-3, 100)
    sampling_experiment.run(sampling_algorithm, noise, sigma, "DC")


if __name__ == "__main__":
    main()
