import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import SecondOrderECirreFDMSolverExplicitApprox
from src.params import SecondOrderECirreMechanismFDMParams
from src.voltammetry import CyclicDC

jax.config.update("jax_enable_x64", True)

voltammetry = CyclicDC()

solver = SecondOrderECirreFDMSolverExplicitApprox(voltammetry, h=1e-3, dtheta=1e-2)

params = SecondOrderECirreMechanismFDMParams(
    alpha=jnp.array(0.5),
    K0=jnp.array(1000.0),
    Kplus=jnp.array(100000.0),
    Kminus=jnp.array(1.0),
    dB=jnp.array(1.0),
    dY=jnp.array(1.0),
    dZ=jnp.array(1.0),
    E0=jnp.array(0.0),
)

current = solver.solve(params)

plt.plot(solver.applied_potentials, current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
