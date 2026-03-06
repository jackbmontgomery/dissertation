import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import AdsorptionReactionNewtonFDSolver
from src.params import AdsorptionReactionParams
from src.voltammetry import CyclicAC, CyclicDC

params = AdsorptionReactionParams(
    alpha_sol=jnp.array(0.4),
    K0_sol=jnp.array(1e-3),
    Ef_sol=jnp.array(0.0),
    alpha_ads=jnp.array(0.45),
    K0_ads=jnp.array(5e-1),
    K_A_ads=jnp.array(4.5),
    K_A_des=jnp.array(1.0),
    K_B_ads=jnp.array(1.0),
    K_B_des=jnp.array(1.0),
)


ac = CyclicAC(theta_i=20.0, theta_v=-20.0, sigma=10)
dc = CyclicDC(theta_i=20.0, theta_v=-20.0, sigma=10)

ac_fd_solver = AdsorptionReactionNewtonFDSolver(ac, dtheta=1e-1)
dc_fd_solver = AdsorptionReactionNewtonFDSolver(dc, dtheta=1e-1)

_, ac_current = ac_fd_solver.solve(params)
_, dc_current = dc_fd_solver.solve(params)

plt.plot(dc_fd_solver.applied_potentials, ac_current)
plt.plot(dc_fd_solver.applied_potentials, dc_current)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
