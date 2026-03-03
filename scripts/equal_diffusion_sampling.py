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
        100: jnp.repeat(0.025, 3),
        1000: jnp.repeat(0.05, 3),
    }
}


def main():
    sampling_experiment = EqualDiffusionReactionSamplingExperiment()
    sigma = 100
    noise = 0.1
    seed = 42

    # --- Random-Walk ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=160_000,
        sigma=hyperparams["RW"][sigma],
    )
    sampling_experiment.run(sampling_algorithm, noise, sigma, seed=seed)

    # --- HMC ---
    sampling_algorithm = HMCSamplingAlgorithm(32_000, 1e-2, 1e-2, 100)
    sampling_experiment.run(sampling_algorithm, noise, sigma, seed=seed)


if __name__ == "__main__":
    main()
