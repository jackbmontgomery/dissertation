import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import ElectronReactionSamplingExperiment
from src.sampling import (
    AdditiveStepRandomWalkSamplingAlgorithm,
    HMCSamplingAlgorithm,
    MCHMCSamplingAlgorithm,
)
from src.voltammetry import CyclicDC

"""
25/02 - 8000/200_000 for MCHMC/AdditiveStep gives ~200 seconds of runtime
"""


def main():
    voltammetry = CyclicDC()
    sampling_experiment = ElectronReactionSamplingExperiment()

    # --- Metropolis-Hasting ---
    sampling_algorithm = AdditiveStepRandomWalkSamplingAlgorithm(
        n_samples=200_000,
        sigma=jnp.array([0.005, 0.005, 0.005, 0.005]),
    )
    sampling_experiment.run(sampling_algorithm, voltammetry)

    # --- MCHMC ---
    # sampling_algorithm = MCHMCSamplingAlgorithm(n_samples=8000, step_size=5e-3)
    # sampling_experiment.run(sampling_algorithm, voltammetry)

    # --- HMC ---
    sampling_algorithm = HMCSamplingAlgorithm(80_000, 1e-2, 1e-3, 100)
    sampling_experiment.run(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
