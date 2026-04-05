import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from typing import Literal

import jax.numpy as jnp

from src.fdm import (
    AdsorptionReactionExplicitFDSolver,
    AdsorptionReactionNewtonFDSolver,
    ElectronReactionFDSolver,
    HeterogeneousReactionFDSolver,
)
from src.reaction import AdsorptionReaction, ElectronReaction, HeterogeneousReaction
from src.sampling_experiment import sampling_experiment
from src.voltammetry import CyclicDC

num_chains = multiprocessing.cpu_count()


def main(name: Literal["e", "h", "a"], seed: int = 0, save: bool = True):
    if name == "e":
        reaction = ElectronReaction()
        voltammetry = CyclicDC()
        fd_solver = ElectronReactionFDSolver(voltammetry)

        sampling_experiment(
            reaction,
            fd_solver,
            experimental_noise=0.03,
            rwmh_scale_factor=50.0,
            num_rwmh_samples=160_000,
            num_nuts_samples=16_000,
            seed=seed,
            save=save,
        )

    elif name == "h":
        reaction = HeterogeneousReaction()
        voltammetry = CyclicDC()
        fd_solver = HeterogeneousReactionFDSolver(voltammetry)

        sampling_experiment(
            reaction,
            fd_solver,
            optim_learning_rate=2e-1,
            optim_steps=250,
            rwmh_scale_factor=15.0,
            num_rwmh_samples=160_000,
            num_nuts_samples=12_000,
            seed=seed + 1,
            save=save,
        )

    elif name == "a":
        reaction = AdsorptionReaction()
        voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=40)
        sampling_fd_solver = AdsorptionReactionExplicitFDSolver(voltammetry)
        data_fd_solver = AdsorptionReactionNewtonFDSolver(voltammetry)

        rwmh_kwargs = {"sigma": jnp.repeat(0.025, 9)}
        hmc_kwargs = {"initial_step_size": 1e-3}

        sampling_experiment(
            reaction,
            sampling_fd_solver,
            num_chains,
            num_rwmh_samples=160_000,
            rwmh_kwargs=rwmh_kwargs,
            num_nuts_samples=64_000,
            hmc_kwargs=hmc_kwargs,
            warmup_steps=500,
            data_fd_solver=data_fd_solver,
            seed=seed,
            save=save,
        )

    else:
        raise Exception("Bad reaction choice")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
