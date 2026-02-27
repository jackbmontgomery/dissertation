import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import HeterogeneousReactionFDSolver
from src.params import HeterogenousReactionParams
from src.voltammetry import CyclicDC

voltammetry = CyclicDC()

fdm_solver = HeterogeneousReactionFDSolver(voltammetry)

base_params = HeterogenousReactionParams(
    alpha_1=jnp.array(0.6),
    K0_1=jnp.array(1.0),
    Ef_1=jnp.array(0.0),
    alpha_2=jnp.array(0.6),
    K0_2=jnp.array(1.0),
    Ef_2=jnp.array(0.0),
    dB=jnp.array(1.0),
    dC=jnp.array(1.0),
    dD=jnp.array(1.0),
    K_het=jnp.array(10.0),
)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    nrows=1, ncols=5, figsize=(12, 3), sharex=True, sharey=True
)

# Alpha Varying

alpha2_range = jnp.linspace(0.3, 0.7, 5)
alpha2_params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(alpha2_range, base_params.alpha_1),
    K0_1=jnp.full_like(alpha2_range, base_params.K0_1),
    Ef_1=jnp.full_like(alpha2_range, base_params.Ef_1),
    alpha_2=alpha2_range,
    K0_2=jnp.full_like(alpha2_range, base_params.K0_2),
    Ef_2=jnp.full_like(alpha2_range, base_params.Ef_2),
    dB=jnp.full_like(alpha2_range, base_params.dB),
    dC=jnp.full_like(alpha2_range, base_params.dC),
    dD=jnp.full_like(alpha2_range, base_params.dD),
    K_het=jnp.full_like(alpha2_range, base_params.K_het),
)


currents = vmap(fdm_solver.solve)(alpha2_params)

for val, current in zip(alpha2_range, currents):
    # ax1.plot(fdm_solver.applied_potentials, current, color="C0")
    ax1.plot(fdm_solver.applied_potentials, current, label=val)

ax1.xaxis.set_inverted(True)
ax1.yaxis.set_inverted(True)
ax1.set_ylabel(r"$J$", fontsize=16)
ax1.set_xlabel(r"$\theta$", fontsize=16)
ax1.set_title(r"$\alpha^{(2)}$", fontsize=20)
ax1.legend()

# K0 Varying
K2_0_range = jnp.array([1.0, 5.0, 10.0, 20.0, 40.0, 50.0])
K2_0_params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(K2_0_range, base_params.alpha_1),
    K0_1=jnp.full_like(K2_0_range, base_params.K0_1),
    Ef_1=jnp.full_like(K2_0_range, base_params.Ef_1),
    alpha_2=jnp.full_like(K2_0_range, base_params.alpha_2),
    K0_2=K2_0_range,
    Ef_2=jnp.full_like(K2_0_range, base_params.Ef_2),
    dB=jnp.full_like(K2_0_range, base_params.dB),
    dC=jnp.full_like(K2_0_range, base_params.dC),
    dD=jnp.full_like(K2_0_range, base_params.dD),
    K_het=jnp.full_like(K2_0_range, base_params.K_het),
)

currents = vmap(fdm_solver.solve)(K2_0_params)

for val, current in zip(K2_0_range, currents):
    ax2.plot(fdm_solver.applied_potentials, current, label=f"{val:.0f}")
ax2.xaxis.set_inverted(True)
ax2.yaxis.set_inverted(True)
ax2.set_title(r"$K^{(2)}_0$", fontsize=20)
ax2.set_xlabel(r"$\theta$", fontsize=16)
ax2.legend()

# dC Varying
dC_range = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
dC_params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(dC_range, base_params.alpha_1),
    K0_1=jnp.full_like(dC_range, base_params.K0_1),
    Ef_1=jnp.full_like(dC_range, base_params.Ef_1),
    alpha_2=jnp.full_like(dC_range, base_params.alpha_2),
    K0_2=jnp.full_like(dC_range, base_params.K0_2),
    Ef_2=jnp.full_like(dC_range, base_params.Ef_2),
    dB=jnp.full_like(dC_range, base_params.dB),
    dC=dC_range,
    dD=jnp.full_like(dC_range, base_params.dD),
    K_het=jnp.full_like(dC_range, base_params.K_het),
)

currents = vmap(fdm_solver.solve)(dC_params)

for val, current in zip(dC_range, currents):
    ax3.plot(fdm_solver.applied_potentials, current, label=val)

ax3.xaxis.set_inverted(True)
ax3.yaxis.set_inverted(True)
ax3.set_title(r"$d_C$", fontsize=20)
ax3.set_xlabel(r"$\theta$", fontsize=16)
ax3.legend()


# dD Varying
dD_range = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
dD_params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(dD_range, base_params.alpha_1),
    K0_1=jnp.full_like(dD_range, base_params.K0_1),
    Ef_1=jnp.full_like(dD_range, base_params.Ef_1),
    alpha_2=jnp.full_like(dD_range, base_params.alpha_2),
    K0_2=jnp.full_like(dD_range, base_params.K0_2),
    Ef_2=jnp.full_like(dD_range, base_params.Ef_2),
    dB=jnp.full_like(dD_range, base_params.dB),
    dC=jnp.full_like(dD_range, base_params.dC),
    dD=dD_range,
    K_het=jnp.full_like(dD_range, base_params.K_het),
)

currents = vmap(fdm_solver.solve)(dD_params)

for val, current in zip(dD_range, currents):
    ax4.plot(fdm_solver.applied_potentials, current, label=val)

ax4.xaxis.set_inverted(True)
ax4.yaxis.set_inverted(True)
ax4.set_title(r"$d_D$", fontsize=20)
ax4.set_xlabel(r"$\theta$", fontsize=16)
ax4.legend()

# Khet Varying
K_het_range = jnp.array([1.0, 5.0, 10.0, 20.0, 40.0, 50.0])
K_het_params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(K2_0_range, base_params.alpha_1),
    K0_1=jnp.full_like(K2_0_range, base_params.K0_1),
    Ef_1=jnp.full_like(K2_0_range, base_params.Ef_1),
    alpha_2=jnp.full_like(K2_0_range, base_params.alpha_2),
    K0_2=jnp.full_like(K2_0_range, base_params.K0_2),
    Ef_2=jnp.full_like(K2_0_range, base_params.Ef_2),
    dB=jnp.full_like(K2_0_range, base_params.dB),
    dC=jnp.full_like(K2_0_range, base_params.dC),
    dD=jnp.full_like(K2_0_range, base_params.dD),
    K_het=K_het_range,
)

currents = vmap(fdm_solver.solve)(K_het_params)

for val, current in zip(K_het_range, currents):
    ax5.plot(fdm_solver.applied_potentials, current, label=f"{val:.0f}")

ax5.xaxis.set_inverted(True)
ax5.yaxis.set_inverted(True)
ax5.set_title(r"$K_{\text{het}}$", fontsize=20)
ax5.set_xlabel(r"$\theta$", fontsize=16)
ax5.legend()

plt.tight_layout()
plt.show()
