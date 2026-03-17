from src.fdm import (
    AdsorptionReactionExplicitFDSolver,
    ElectronReactionFDSolver,
    HeterogeneousReactionFDSolver,
)
from src.optimisation_experiment import optimisation_experiment
from src.reaction import ElectronReaction, HeterogeneousReaction
from src.voltammetry import CyclicDC

seed = 0

# %%  ---------- Electron Reaction ----------

voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)
reaction = ElectronReaction()

adam_params = {"learning_rate": 1e-2}
cmaes_param = {"population_size": 4}

optimisation_experiment(
    reaction,
    fd_solver,
    num_iterations=200,
    num_params=32,
    experimental_noise=0.5,
    cmaes_params=cmaes_param,
    adam_params=adam_params,
    seed=seed,
)

# %%  ---------- Heterogeneous Reaction ----------

# voltammetry = CyclicDC()
# fd_solver = HeterogeneousReactionFDSolver(voltammetry)
# reaction = HeterogeneousReaction()
#
# adam_params = {"learning_rate": 1e-1}
# cmaes_param = {"population_size": 4}
#
# optimisation_experiment(
#     reaction,
#     fd_solver,
#     num_iterations=200,
#     num_params=8,
#     cmaes_params=cmaes_param,
#     adam_params=adam_params,
#     seed=seed,
# )
#
# # %%  ---------- Adsorption Reaction ----------
#
# voltammetry = CyclicDC()
# fd_solver = AdsorptionReactionExplicitFDSolver(voltammetry)
# reaction = AdsorptionReaction()
#
# adam_params = {"learning_rate": 1e-1}
# cmaes_param = {"population_size": 4}
#
# optimisation_experiment(
#     reaction,
#     fd_solver,
#     num_iterations=200,
#     num_params=8,
#     cmaes_params=cmaes_param,
#     adam_params=adam_params,
#     seed=seed,
# )
