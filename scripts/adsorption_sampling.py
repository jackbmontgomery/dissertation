import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import AdsorptionSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    HMCSamplingAlgorithm,
)

hyperparams = {
    "RW": {
        10: jnp.repeat(0.05, 9),
        100: jnp.repeat(0.025, 9),
        1000: jnp.repeat(0.005, 9),
    }
}


def main():
    sampling_experiment = AdsorptionSamplingExperiment()
    sigma = 10
    noise = 0.01

    # --- Metropolis-Hasting ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=240_000,
        sigma=hyperparams["RW"][sigma],
    )
    sampling_experiment.run(sampling_algorithm, noise, sigma)

    # --- HMC ---
    # sampling_algorithm = HMCSamplingAlgorithm(4_000, 1e-1, 1e-3, 200)
    # sampling_experiment.run(sampling_algorithm, noise, sigma)


if __name__ == "__main__":
    main()
