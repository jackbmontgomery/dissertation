import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp

from src.experiment import e_reaction_sampling_experiment
from src.sampling import (
    HMCSamplingAlgorithm,
    MetropolisHastingsSamplingAlgorithm,
)
from src.voltammetry import CyclicDC


def main():
    voltammetry = CyclicDC()

    # --- Metropolis-Hasting ---
    sampling_algorithm = MetropolisHastingsSamplingAlgorithm(
        n_samples=160_000,
        sigma=jnp.array([0.005, 0.005, 0.005, 0.005]),
    )
    e_reaction_sampling_experiment(sampling_algorithm, voltammetry)

    # --- HMC ---
    sampling_algorithm = HMCSamplingAlgorithm(
        n_samples=8000, learning_rate=1e-2, initial_step_size=1e-2, warmup_steps=100
    )
    e_reaction_sampling_experiment(sampling_algorithm, voltammetry)


if __name__ == "__main__":
    main()
