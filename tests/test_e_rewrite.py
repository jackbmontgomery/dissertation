import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.voltammetry import CyclicDC

voltammetry = CyclicDC()

solver = EMechanismFDMSolver(voltammetry)

params = EMechanismFDMParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(1000.0),
    dB=jnp.array(1.0),
    E0=jnp.array(0.0),
)

current = solver.solve(params)

plt.plot(solver.applied_potentials, current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
