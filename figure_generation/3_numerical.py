import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import vmap

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2.0)

# %% Analytical results

voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

params = ElectronReactionParams(
    alpha=jnp.array(0.5),
    K0=jnp.array(1000.0),
    thetaf=jnp.array(0.0),
)

current = fd_solver.solve(params)
ax1.plot(fd_solver.applied_potentials, current)

max_current = -0.446 * jnp.sqrt(voltammetry.sigma)
ax1.axhline(y=max_current, c="C3", linestyle="--", label="Analytical")

params = ElectronReactionParams(
    alpha=jnp.array(0.7),
    K0=jnp.array(1.0),
    thetaf=jnp.array(0.0),
)

current = fd_solver.solve(params)


ax2.plot(fd_solver.applied_potentials, current, label="Numerical")

max_current = -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(voltammetry.sigma)
ax2.axhline(y=max_current, c="C3", linestyle="--", label="Analytical")
max_current_position = (
    jnp.log(params.K0 / jnp.sqrt(params.alpha * voltammetry.sigma)) - 0.78
) / params.alpha
ax2.axvline(x=max_current_position, c="C3", linestyle="--")

ax1.set_ylabel(r"$J$")
ax1.set_xlabel(r"$\theta$")
ax2.set_xlabel(r"$\theta$")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/3-analytical.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline quasi-reversible reaction

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(20.0), thetaf=jnp.array(0.5)
)

voltammetry = CyclicDC()

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

plt.savefig("./manuscript/figures/3-parameter-effect-quasi.png", dpi=1000)
plt.show()

# %% Effect of parameters using a baseline reversible reaction

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

rev_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(200.0), thetaf=jnp.array(0.5)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, rev_params.K0),
    thetaf=jnp.full_like(alpha_range, rev_params.thetaf),
)

alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    ax1.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

ax1.xaxis.set_inverted(True)
ax1.yaxis.set_inverted(True)

ax1.set_title(r"$\alpha$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()

K0_range = jnp.array([200.0, 500.0, 1000.0])
K0_params = ElectronReactionParams(
    alpha=jnp.full_like(K0_range, rev_params.alpha),
    K0=K0_range,
    thetaf=jnp.full_like(K0_range, rev_params.thetaf),
)

K0_currents = vmap(fd_solver.solve)(K0_params)
for val, current in zip(K0_range, K0_currents):
    ax2.plot(fd_solver.applied_potentials, current, label=f"{val:.0f}")

ax2.xaxis.set_inverted(True)
ax2.yaxis.set_inverted(True)

ax2.set_title(r"$K_0$")
ax2.set_xlabel(r"$\theta$")
ax2.legend()

plt.tight_layout()
plt.savefig("./manuscript/figures/3-alpha-K0-effect-reversible.png", dpi=1000)
plt.show()


# %% Choice of discritisation

voltammetry = CyclicDC()
params = ElectronReactionParams(
    alpha=jnp.array(0.7),
    K0=jnp.array(1.0),
    thetaf=jnp.array(0.0),
)

base_dtheta = 1e-4
base_h0 = 1e-10
fd_solver = ElectronReactionFDSolver(voltammetry, h0=base_h0, dtheta=base_dtheta)
base_current = fd_solver.solve(params).block_until_ready()
base_time = jnp.arange(len(base_current)) * base_dtheta

h0_range = jnp.power(10.0, jnp.arange(-9, -2))
dtheta_range = [2e-4, 5e-4, 8e-4, 1e-3, 2e-3, 5e-3, 1e-2]

for dtheta in dtheta_range:
    dtheta_vals = []
    coarse_time = jnp.arange(int(base_time[-1] / dtheta) + 1) * dtheta
    base_interp = jnp.interp(coarse_time, base_time, base_current)
    for h0 in h0_range:
        fd_solver = ElectronReactionFDSolver(voltammetry, h0=h0, dtheta=dtheta)
        current = fd_solver.solve(params).block_until_ready()
        n = min(len(current), len(base_interp))
        rel_l2 = jnp.linalg.norm(current[:n] - base_interp[:n]) / jnp.linalg.norm(
            base_interp[:n]
        )
        dtheta_vals.append(float(rel_l2))

    plt.plot(h0_range, dtheta_vals, label=dtheta)


plt.yscale("log")
plt.xscale("log")
plt.xlabel("h0")
plt.legend()
plt.tight_layout()
plt.show()

# %%

voltammetry = CyclicDC()
params = ElectronReactionParams(
    alpha=jnp.array(0.7),
    K0=jnp.array(1.0),
    thetaf=jnp.array(0.0),
)

# Reference solution: very fine in both space and time
ref_dtheta = 1e-5
ref_h0 = 1e-10
ref_solver = ElectronReactionFDSolver(voltammetry, h0=ref_h0, dtheta=ref_dtheta)
ref_current = ref_solver.solve(params).block_until_ready()
ref_time = jnp.arange(len(ref_current)) * ref_dtheta

# Fix h0 in the flat region, sweep dtheta
h0_fixed = 1e-8
dtheta_range = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
errors = []

for dtheta in dtheta_range:
    solver = ElectronReactionFDSolver(voltammetry, h0=h0_fixed, dtheta=dtheta)
    current = solver.solve(params).block_until_ready()
    coarse_time = jnp.arange(len(current)) * dtheta
    ref_interp = jnp.interp(coarse_time, ref_time, ref_current)
    rel_l2 = float(jnp.linalg.norm(current - ref_interp) / jnp.linalg.norm(ref_interp))
    errors.append(rel_l2)
    print(f"dtheta = {dtheta:.4f}, rel L2 error = {rel_l2:.2e}")

# Compute empirical convergence order between successive pairs
print("\nConvergence order:")
for i in range(1, len(dtheta_range)):
    ratio = errors[i] / errors[i - 1]
    dt_ratio = dtheta_range[i] / dtheta_range[i - 1]
    order = jnp.log(ratio) / jnp.log(dt_ratio)
    print(
        f"dtheta {dtheta_range[i - 1]:.4f} -> {dtheta_range[i]:.4f}: order = {float(order):.2f}"
    )
