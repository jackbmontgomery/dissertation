import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2.0)

# %% Applied Potential and Voltammagram

true_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(1e6), thetaf=jnp.array(0.0)
)

voltammetry = CyclicDC(theta_i=10.0, theta_v=-10.0, sigma=1000.0)

fd_solver = ElectronReactionFDSolver(voltammetry)

current = fd_solver.solve(true_params)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

ax1.plot(fd_solver.applied_potentials)
ax1.yaxis.set_inverted(True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Applied Potential")
ax1.set_xlabel("Time")

ax2.plot(fd_solver.applied_potentials, current)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.xaxis.set_inverted(True)
ax2.yaxis.set_inverted(True)
ax2.set_ylabel("Current")
ax2.set_xlabel("Applied Potential")
plt.tight_layout()
plt.savefig("./manuscript/figures/2-voltammagram.png", dpi=1000)
plt.show()
