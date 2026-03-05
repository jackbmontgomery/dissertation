import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import ElectronSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    HMCSamplingAlgorithm,
)

hyperparams = {
    "RW": {
        100: jnp.repeat(0.01, 3),
        1000: jnp.repeat(0.01, 3),
    }
}


def main():
    sampling_experiment = ElectronSamplingExperiment()
    sigma = 1000
    noise = 0.25
    seed = 42

    # --- Random-Walk ---
    # sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
    #     n_samples=160_000,
    #     sigma=hyperparams["RW"][sigma],
    # )
    # sampling_experiment.run(sampling_algorithm, noise, sigma, seed)

    # --- HMC ---
    sampling_algorithm = HMCSamplingAlgorithm(16_000, 1e-1, 1e-3, 500)
    sampling_experiment.run(sampling_algorithm, noise, sigma, seed)


if __name__ == "__main__":
    main()
