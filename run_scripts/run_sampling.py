import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax.numpy as jnp

from src.fdm import (
    ElectronReactionFDSolver,
    HeterogeneousReactionFDSolver,
)
from src.reaction import ElectronReaction, HeterogeneousReaction
from src.sampling_experiment import sampling_experiment
from src.voltammetry import CyclicDC

seed = 0

num_chains = multiprocessing.cpu_count()

# %%  ---------- Electron Reaction ----------

reaction = ElectronReaction()
voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)

rwmh_kwargs = {"sigma": jnp.repeat(0.01, 3)}
hmc_kwargs = {"initial_step_size": 1e-3}

sampling_experiment(
    reaction,
    fd_solver,
    num_chains,
    num_rwmh_samples=160_000,
    rwmh_kwargs=rwmh_kwargs,
    num_hmc_samples=32_000,
    hmc_kwargs=hmc_kwargs,
    seed=seed,
)


# %%  ---------- Heterogeneous Reaction ----------

# reaction = HeterogeneousReaction()
# voltammetry = CyclicDC()
# fd_solver = HeterogeneousReactionFDSolver(voltammetry)
#
# rwmh_kwargs = {"sigma": jnp.repeat(0.01, 7)}
# hmc_kwargs = {"initial_step_size": 1e-3}
#
# sampling_experiment(
#     reaction,
#     fd_solver,
#     num_chains,
#     num_rwmh_samples=160_000,
#     rwmh_kwargs=rwmh_kwargs,
#     num_hmc_samples=32_000,
#     hmc_kwargs=hmc_kwargs,
#     seed=seed,
# )
