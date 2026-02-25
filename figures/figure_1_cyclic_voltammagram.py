import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from jax import vmap

from src.fdm import (
    ECirreMechanismFDMSolver,
    EMechanismFDMSolver,
    SecondOrderECirreFDMSolverBackwardImplicit,
    SecondOrderECirreFDMSolverExplicit,
)
from src.params import (
    ECirreMechanismFDMParams,
    EMechanismFDMParams,
    SecondOrderECirreMechanismFDMParams,
)
from src.plotting import plot_e_histograms, plot_ec_irre_histograms
from src.voltammetry import CyclicDC

plt.style.use("science")

voltammetry = CyclicDC()

fdm_solver = EMechanismFDMSolver(voltammetry)

params = EMechanismFDMParams(
    alpha=jnp.array(0.7),
    K0=jnp.array(1.0),
    E0=jnp.array(0.0),
    dB=jnp.array(1.0),
)

current = fdm_solver.solve(params)

plt.figure(figsize=(6, 4))
plt.plot(fdm_solver.applied_potentials, current)
plt.axhline(
    y=float(-0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(voltammetry.sigma)),
    linestyle="--",
    c="red",
)
plt.axvline(
    x=(jnp.log(params.K0 / jnp.sqrt(params.alpha * voltammetry.sigma)) - 0.78)
    / params.alpha,
    linestyle="--",
    c="red",
)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.ylabel(r"$J$", fontsize=16)
plt.xlabel(r"$\theta$", fontsize=16)
plt.tight_layout()
plt.show()
