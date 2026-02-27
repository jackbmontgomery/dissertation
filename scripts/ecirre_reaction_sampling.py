import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import ECirreSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    MCHMCSamplingAlgorithm,
)
from src.voltammetry import CyclicDC


def main():
    voltammetry = CyclicDC()
    sampling_experiment = ECirreSamplingExperiment()

    # --- Metropolis-Hasting ---
    # sampling_algorithm = MetropolisHastingsSamplingAlgorithm(
    #     n_samples=200_000,
    #     sigma=jnp.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005]),
    # )
    # sampling_experiment.run(sampling_algorithm, voltammetry)

    # --- MCHMC ---
    sampling_algorithm = MCHMCSamplingAlgorithm(n_samples=4000, step_size=5e-3)
    sampling_experiment.run(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
