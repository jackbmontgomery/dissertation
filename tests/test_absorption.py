import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import AbsorptionReactionNewtonDFSolver
from src.params import AbsorptionReactionParams
from src.voltammetry import CyclicDC

params = AbsorptionReactionParams(
    alpha_sol=jnp.array(0.4),
    K0_sol=jnp.array(5.0),
    Ef_sol=jnp.array(1.0),
    alpha_abs=jnp.array(0.45),
    K0_abs=jnp.array(0.5),
    K_A_abs=jnp.array(4.5),
    K_A_des=jnp.array(1.0),
    K_B_abs=jnp.array(1.0),
    K_B_des=jnp.array(1.0),
    dB=jnp.array(0.8),
)

voltammetry = CyclicDC()

fdm_solver = AbsorptionReactionNewtonDFSolver(voltammetry)

current = fdm_solver.solve(params)

plt.plot(fdm_solver.applied_potentials, current)
plt.show()
