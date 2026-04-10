import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from typing import Literal

from src.fdm import (
    AdsorptionReactionExplicitFDSolver,
    AdsorptionReactionNewtonFDSolver,
    ElectronReactionFDSolver,
    HeterogeneousReactionFDSolver,
)
from src.reaction import AdsorptionReaction, ElectronReaction, HeterogeneousReaction
from src.sampling_experiment import sampling_experiment
from src.voltammetry import CyclicAC, CyclicDC

num_chains = multiprocessing.cpu_count()


def main(name: Literal["e", "h", "a"], seed: int = 0, save: bool = True):
    if name == "e":
        reaction = ElectronReaction()
        voltammetry = CyclicDC()
        fd_solver = ElectronReactionFDSolver(voltammetry)

        sampling_experiment(
            reaction,
            fd_solver,
            optim_steps=50,
            experimental_noise=0.02,
            num_rwmh_samples=20_000,
            num_nuts_samples=2_000,
            seed=seed,
            save=save,
        )

    elif name == "h":
        if seed == 1:
            voltammetry = CyclicAC()
        else:
            voltammetry = CyclicDC()

        reaction = HeterogeneousReaction()
        fd_solver = HeterogeneousReactionFDSolver(voltammetry)

        sampling_experiment(
            reaction,
            fd_solver,
            optim_learning_rate=2e-1,
            optim_steps=250,
            num_rwmh_samples=32_000,
            num_nuts_samples=2_400,
            seed=seed,
            save=save,
        )

    elif name == "a":
        reaction = AdsorptionReaction()
        voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=50)
        sampling_fd_solver = AdsorptionReactionExplicitFDSolver(voltammetry)
        data_fd_solver = AdsorptionReactionNewtonFDSolver(voltammetry)

        sampling_experiment(
            reaction,
            sampling_fd_solver,
            data_fd_solver=data_fd_solver,
            optim_learning_rate=2e-1,
            optim_steps=1000,
            num_rwmh_samples=160_000,
            num_nuts_samples=8_000,
            seed=seed,
            save=save,
        )

    else:
        raise Exception("Bad reaction choice")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
