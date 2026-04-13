from math import floor, log
from typing import Literal

from src.fdm import (
    AdsorptionReactionExplicitFDSolver,
    ElectronReactionFDSolver,
    HeterogeneousReactionFDSolver,
)
from src.optimisation_experiment import optimisation_experiment
from src.reaction import AdsorptionReaction, ElectronReaction, HeterogeneousReaction
from src.voltammetry import CyclicDC


def main(
    name: Literal["e", "h", "a"], noise: float = 0.02, seed: int = 0, save: bool = True
):
    if name == "e":
        voltammetry = CyclicDC()
        fd_solver = ElectronReactionFDSolver(voltammetry)
        reaction = ElectronReaction()

        adam_params = {"learning_rate": 2e-1}
        cmaes_param = {"population_size": 4 + floor(3 * log(reaction.parameter_dim))}

        optimisation_experiment(
            reaction,
            fd_solver,
            budget=210,
            num_params=32,
            noise_percentage=noise,
            cmaes_params=cmaes_param,
            adam_params=adam_params,
            seed=seed,
            save=save,
        )

    elif name == "h":
        voltammetry = CyclicDC()
        fd_solver = HeterogeneousReactionFDSolver(voltammetry)
        reaction = HeterogeneousReaction()

        adam_params = {"learning_rate": 1e-1}
        cmaes_param = {"population_size": 4 + floor(3 * log(reaction.parameter_dim))}

        optimisation_experiment(
            reaction,
            fd_solver,
            budget=400,
            num_params=32,
            noise_percentage=noise,
            cmaes_params=cmaes_param,
            adam_params=adam_params,
            seed=seed,
            save=save,
        )

    elif name == "a":
        reaction = AdsorptionReaction()
        voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=10)
        fd_solver = AdsorptionReactionExplicitFDSolver(voltammetry)

        adam_params = {"learning_rate": 2e-1}
        cmaes_param = {"population_size": 4 + floor(3 * log(reaction.parameter_dim))}

        optimisation_experiment(
            reaction,
            fd_solver,
            budget=400,
            num_params=32,
            noise_percentage=noise,
            cmaes_params=cmaes_param,
            adam_params=adam_params,
            seed=seed,
            save=save,
        )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
