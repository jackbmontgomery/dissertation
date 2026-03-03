import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import (
    EqualDiffusionReactionSamplingExperiment,
)
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    HMCSamplingAlgorithm,
)

hyperparams = {
    "RW": {
        100: jnp.repeat(0.2, 2),
        1000: jnp.repeat(0.05, 2),
    }
}


def main():
    sampling_experiment = EqualDiffusionReactionSamplingExperiment()
    sigma = 100
    noise = 0.1

    # --- Random-Walk ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=80_000,
        sigma=hyperparams["RW"][sigma],
    )
    sampling_experiment.run(sampling_algorithm, noise, sigma)

    # --- HMC ---
    # sampling_algorithm = HMCSamplingAlgorithm(8_000, 1e-2, 1e-2, 100)
    # sampling_experiment.run(sampling_algorithm, noise, sigma)


if __name__ == "__main__":
    main()
