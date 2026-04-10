import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import vmap

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2.0)

save = False

# %% Analytical results
voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=100)
fd_solver = ElectronReactionFDSolver(voltammetry)

params = ElectronReactionParams(
    alpha=jnp.array(0.7),
    K0=jnp.array(1.0),
    thetaf=jnp.array(0.0),
)

current = fd_solver.solve(params)

max_current = -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(voltammetry.sigma)
plt.axhline(y=max_current, c="C1", linestyle="--", label="Analytical")
max_current_position = (
    jnp.log(params.K0 / jnp.sqrt(params.alpha * voltammetry.sigma)) - 0.78
) / params.alpha
plt.axvline(x=max_current_position, c="C1", linestyle="--")

plt.plot(fd_solver.applied_potentials, current, label="Numerical")
plt.ylabel(r"$J$")
plt.xlabel(r"$\theta$")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
if save:
    plt.savefig("./manuscript/figures/3-analytical.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline quasi-reversible reaction

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(10.0), thetaf=jnp.array(0.5)
)

voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=100)

fd_solver = ElectronReactionFDSolver(voltammetry)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, rev_params.K0),
    thetaf=jnp.full_like(alpha_range, rev_params.thetaf),
)

alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    ax0.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

K0_range = jnp.array([1.0, 10.0, 50.0])

K0_params = ElectronReactionParams(
    alpha=jnp.full_like(K0_range, rev_params.alpha),
    K0=K0_range,
    thetaf=jnp.full_like(K0_range, rev_params.thetaf),
)

K0_currents = vmap(fd_solver.solve)(K0_params)
for val, current in zip(K0_range, K0_currents):
    ax1.plot(fd_solver.applied_potentials, current, label=f"{val:.0f}")

Ef_range = jnp.array([-1.0, 0.0, 1.0])

Ef_params = ElectronReactionParams(
    alpha=jnp.full_like(Ef_range, rev_params.alpha),
    K0=jnp.full_like(Ef_range, rev_params.K0),
    thetaf=Ef_range,
)

Ef_currents = vmap(fd_solver.solve)(Ef_params)
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

if save:
    plt.savefig("./manuscript/figures/3-parameter-effect-quasi.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline reversible reaction

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(100.0), thetaf=jnp.array(0.5)
)

voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=100)

fd_solver = ElectronReactionFDSolver(voltammetry)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, rev_params.K0),
    thetaf=jnp.full_like(alpha_range, rev_params.thetaf),
)

alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    plt.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

plt.title(r"$\alpha$")
plt.ylabel(r"$J$")
plt.xlabel(r"$\theta$")
plt.legend()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
if save:
    plt.savefig("./manuscript/figures/3-alpha-effect-reversible.png", dpi=1000)
plt.show()
