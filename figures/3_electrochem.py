import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from jax import vmap

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

key = jr.key(0)

# %% Applied Potential and Voltammagram

true_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(1e6), Ef=jnp.array(0.0)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

_, current = fd_solver.solve(true_params)

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
plt.savefig("./write_up/figures/applied-potential-and-voltammagram.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline quasi-reversible reaction

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(10.0), Ef=jnp.array(0.5)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, rev_params.K0),
    Ef=jnp.full_like(alpha_range, rev_params.Ef),
)

_, alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    ax0.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

K0_range = jnp.array([1.0, 10.0, 50.0])

K0_params = ElectronReactionParams(
    alpha=jnp.full_like(K0_range, rev_params.alpha),
    K0=K0_range,
    Ef=jnp.full_like(K0_range, rev_params.Ef),
)

_, K0_currents = vmap(fd_solver.solve)(K0_params)
for val, current in zip(K0_range, K0_currents):
    ax1.plot(fd_solver.applied_potentials, current, label=f"{val:.0f}")

Ef_range = jnp.array([-1.0, 0.0, 1.0])

Ef_params = ElectronReactionParams(
    alpha=jnp.full_like(Ef_range, rev_params.alpha),
    K0=jnp.full_like(Ef_range, rev_params.K0),
    Ef=Ef_range,
)

_, Ef_currents = vmap(fd_solver.solve)(Ef_params)
for val, current in zip(Ef_range, Ef_currents):
    ax2.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

ax0.set_title(r"$\alpha$")
ax0.set_ylabel(r"$J$")
ax0.set_xlabel(r"$\theta$")
ax0.legend()

ax1.set_title(r"$K_0$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()

ax2.set_title(r"$\theta_f$")
ax2.set_xlabel(r"$\theta$")
ax2.legend()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./write_up/figures/3-parameter-effect-quasi.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline reversible reaction

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(200.0), Ef=jnp.array(0.5)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, rev_params.K0),
    Ef=jnp.full_like(alpha_range, rev_params.Ef),
)

_, alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    plt.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

plt.title(r"$\alpha$")
plt.ylabel(r"$J$")
plt.xlabel(r"$\theta$")
plt.legend()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./write_up/figures/3-alpha-effect-reversible.png", dpi=1000)
plt.show()
