import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import MicroElectrodeFDMSolver
from src.params import MacroElectrodeParams
from src.voltammetry import LinearSweepDC

sigma = 20000.0
h0 = 1e-4
omega = 1.1
dtheta = 0.05

voltammetry = LinearSweepDC(sigma=sigma)
fdm_solver = MicroElectrodeFDMSolver(voltammetry, h0, omega, dtheta)

params = MacroElectrodeParams(
    alpha=jnp.array(0.7), kappa=jnp.array(1000.0), epsilon=jnp.array(0.0)
)

current = fdm_solver.solve(params)
# max_current_estimate = jnp.min(current)

p = jnp.sqrt(sigma)
max_current = 0.439 * p + 0.713 * p**0.108 + (0.614 * p) / (1 + 10.9 * p**2)
# assert jnp.isclose(max_current_estimate, -1 * max_current, rtol=0.02)

plt.plot(fdm_solver.applied_potentials, current)
plt.axhline(y=-1 * max_current)
plt.show()
