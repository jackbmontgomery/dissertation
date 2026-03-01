import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import (
    AdsorptionReactionBackwardImplicitFDSolver,
    AdsorptionReactionExplicitFDSolver,
    AdsorptionReactionNewtonFDSolver,
)
from src.params import AdsorptionReactionParams
from src.voltammetry import CyclicDC

jax.config.update("jax_enable_x64", True)

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
    dB=jnp.array(1.0),
)

voltammetry = CyclicDC(sigma=10, theta_i=20, theta_v=-20)
explicit_fd_solver = AdsorptionReactionExplicitFDSolver(
    voltammetry, h0=1e-6, dtheta=1e-1
)

newton_fd_solver = AdsorptionReactionNewtonFDSolver(voltammetry, h0=1e-6, dtheta=1e-1)
backward_fd_solver = AdsorptionReactionBackwardImplicitFDSolver(
    voltammetry, h0=1e-6, dtheta=1e-1
)

explicit_current = explicit_fd_solver.solve(params)
newton_current = newton_fd_solver.solve(params)
backward_current = backward_fd_solver.solve(params)

plt.plot(explicit_fd_solver.applied_potentials, explicit_current, label="Explicit")
plt.plot(
    newton_fd_solver.applied_potentials, newton_current, label="Newton", linestyle="--"
)
plt.plot(
    backward_fd_solver.applied_potentials, backward_current, label="Backward Implicit"
)

plt.legend()
plt.show()
