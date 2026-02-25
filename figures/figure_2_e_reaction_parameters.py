import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.voltammetry import CyclicDC

voltammetry = CyclicDC()

fdm_solver = EMechanismFDMSolver(voltammetry)

base_params = EMechanismFDMParams(
    alpha=jnp.array(0.6), K0=jnp.array(1.0), E0=jnp.array(2.0), dB=jnp.array(0.5)
)

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=(12, 5), sharex=True, sharey=True
)

# Alpha Varying

alpha_range = jnp.linspace(0.3, 0.7, 5)
alpha_params = EMechanismFDMParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, base_params.K0),
    E0=jnp.full_like(alpha_range, base_params.E0),
    dB=jnp.full_like(alpha_range, base_params.dB),
)


currents = vmap(fdm_solver.solve)(alpha_params)

for val, current in zip(alpha_range, currents):
    ax1.plot(fdm_solver.applied_potentials, current, color="C0")
    # ax1.plot(fdm_solver.applied_potentials, current, label=val)

ax1.xaxis.set_inverted(True)
ax1.yaxis.set_inverted(True)
ax1.set_ylabel(r"$J$", fontsize=16)
ax1.set_xlabel(r"$\theta$", fontsize=16)
ax1.set_title(r"$\alpha$", fontsize=20)
# ax1.legend()

ax1.annotate(
    "",
    xy=(-6, -13),
    xytext=(-10, -10),
    arrowprops=dict(arrowstyle="->", linewidth=1),
)

ax1.annotate(
    "",
    xy=(10, 5.5),
    xytext=(6, 6),
    arrowprops=dict(arrowstyle="->", linewidth=1),
)

# K0 Varying
K0_range = jnp.array([1.0, 5.0, 10.0, 20.0, 40.0, 50.0])
K0_params = EMechanismFDMParams(
    alpha=jnp.full_like(K0_range, base_params.alpha),
    E0=jnp.full_like(K0_range, base_params.E0),
    dB=jnp.full_like(K0_range, base_params.dB),
    K0=K0_range,
)

currents = vmap(fdm_solver.solve)(K0_params)

for val, current in zip(K0_range, currents):
    ax2.plot(fdm_solver.applied_potentials, current, label=f"{val:.0f}")
ax2.xaxis.set_inverted(True)
ax2.yaxis.set_inverted(True)
ax2.set_title(r"$K_0$", fontsize=20)
ax2.set_xlabel(r"$\theta$", fontsize=16)
ax2.legend()

# dB Varying
dB_range = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])

dB_params = EMechanismFDMParams(
    alpha=jnp.full_like(dB_range, base_params.alpha),
    E0=jnp.full_like(dB_range, base_params.E0),
    dB=dB_range,
    K0=jnp.full_like(dB_range, base_params.K0),
)

currents = vmap(fdm_solver.solve)(dB_params)

for val, current in zip(dB_range, currents):
    ax3.plot(fdm_solver.applied_potentials, current, label=val)

ax3.xaxis.set_inverted(True)
ax3.yaxis.set_inverted(True)
ax3.set_title(r"$d_B$", fontsize=20)
ax3.set_xlabel(r"$\theta$", fontsize=16)
ax3.legend()

plt.tight_layout()
plt.show()
