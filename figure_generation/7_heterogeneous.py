import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import jit, vmap

from src.fdm import HeterogeneousReactionFDSolver
from src.params import HeterogenousReactionParams
from src.reaction import HeterogeneousReaction
from src.voltammetry import CyclicAC, CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2)

# %% Analytical Results

voltammetry = CyclicDC()
fd_solver = HeterogeneousReactionFDSolver(voltammetry)
params_1 = HeterogenousReactionParams(
    alpha_1=jnp.array(0.5),
    K0_1=jnp.array(1.0),
    thetaf_1=jnp.array(0.0),
    alpha_2=jnp.array(0.5),
    K0_2=jnp.array(6.0),
    thetaf_2=jnp.array(0.4),
    K_het=jnp.array(0.0),
)

params_2 = HeterogenousReactionParams(
    alpha_1=jnp.array(0.5),
    K0_1=jnp.array(1.0),
    thetaf_1=jnp.array(0.0),
    alpha_2=jnp.array(0.5),
    K0_2=jnp.array(1e6),
    thetaf_2=jnp.array(0.4),
    K_het=jnp.array(1e8),
)

max_current = -0.496 * jnp.sqrt(params_1.alpha_1) * jnp.sqrt(voltammetry.sigma)
plt.axhline(y=max_current, c="C2", linestyle="--", label="Analytical")
plt.axhline(y=2 * max_current, c="C2", linestyle="--", label="Analytical")

max_current_position = (
    jnp.log(params_1.K0_1 / jnp.sqrt(params_1.alpha_1 * voltammetry.sigma)) - 0.78
) / params_1.alpha_1
plt.axvline(x=max_current_position, c="C2", linestyle="--")

current_1 = fd_solver.solve(params_1)
current_2 = fd_solver.solve(params_2)

plt.plot(fd_solver.applied_potentials, current_1)
plt.plot(fd_solver.applied_potentials, current_2)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

# %%

ac_voltammetry = CyclicAC()
dc_voltammetry = CyclicDC()

fd_solver = HeterogeneousReactionFDSolver(ac_voltammetry, dtheta=0.01)
linear_applied_potentials = HeterogeneousReactionFDSolver(
    dc_voltammetry, dtheta=0.01
).applied_potentials

K0_1_range = jnp.array([15.0, 25.0, 50.0])

params = HeterogenousReactionParams(
    alpha_1=jnp.full_like(K0_1_range, 0.5),
    K0_1=K0_1_range,
    thetaf_1=jnp.full_like(K0_1_range, 0.2),
    alpha_2=jnp.full_like(K0_1_range, 0.5),
    K0_2=jnp.full_like(K0_1_range, 6.0),
    thetaf_2=jnp.full_like(K0_1_range, 0.4),
    K_het=jnp.full_like(K0_1_range, 100.0),
)


currents = vmap(fd_solver.solve)(params)


for c in currents:
    plt.plot(linear_applied_potentials, c)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# %% Optimisation Comparison

electron_optim = np.load(
    "./data/optimisation/reaction=HeterogeneousReaction,noise=0.02,seed=0.npz"
)

adam_ld = electron_optim["adam_ld"]
cmaes_ld = electron_optim["cmaes_ld"]
mode_ld = electron_optim["mode_logdensity"]
iterations = jnp.arange(1, adam_ld.shape[1] + 1)

a_ld_mean = jnp.mean(adam_ld, axis=0)
a_ld_std = jnp.std(adam_ld, axis=0)

c_ld_mean = jnp.mean(cmaes_ld, axis=0)
c_ld_std = jnp.std(cmaes_ld, axis=0)

plt.plot(iterations, a_ld_mean, label="ADAM")
plt.fill_between(iterations, a_ld_mean - a_ld_std, a_ld_mean + a_ld_std, alpha=0.5)

plt.plot(iterations, c_ld_mean, label="CMA-ES")
plt.fill_between(iterations, c_ld_mean - c_ld_std, c_ld_mean + c_ld_std, alpha=0.5)

plt.ylim(mode_ld * 5, 300)

plt.axhline(y=mode_ld * 2, linestyle="--", c="C3")
plt.legend()

plt.ylabel("Log Density")
plt.xlabel("Iteration")
plt.tight_layout()
plt.savefig("./manuscript/figures/7-optim.png", dpi=1000)
plt.show()

# %% Sampling: Seed 0

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.02,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

nuts: HeterogenousReactionParams = data["nuts"]
rwmh: HeterogenousReactionParams = data["rwmh"]

true_params: HeterogenousReactionParams = HeterogeneousReaction().true_parameters

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_K1 = fig.add_subplot(gs[0, 1])
ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_a2 = fig.add_subplot(gs[1, 0])
ax_K2 = fig.add_subplot(gs[1, 1])
ax_thetaf2 = fig.add_subplot(gs[1, 2])

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])

options = {"density": True, "bins": 50, "alpha": 0.8, "histtype": "step"}

ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.hist(nuts.alpha_1.flatten(), label="NUTS", **options)
ax_a1.hist(rwmh.alpha_1.flatten(), **options, label="RWMH")
ax_a1.axvline(x=true_params.alpha_1, linestyle="--", color="black", label="True Value")

ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.hist(nuts.alpha_2.flatten(), **options)
ax_a2.hist(rwmh.alpha_2.flatten(), **options)
ax_a2.axvline(x=true_params.alpha_2, linestyle="--", color="black")

ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.hist(nuts.K0_1.flatten(), **options)
ax_K1.hist(rwmh.K0_1.flatten(), **options)
ax_K1.axvline(x=true_params.K0_1, linestyle="--", color="black")

ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.hist(nuts.K0_2.flatten(), **options)
ax_K2.hist(rwmh.K0_2.flatten(), **options)
ax_K2.axvline(x=true_params.K0_2, linestyle="--", color="black")

ax_thetaf1.set_title(r"$E_f^{(1)}$")
ax_thetaf1.hist(nuts.thetaf_1.flatten(), **options)
ax_thetaf1.hist(rwmh.thetaf_1.flatten(), **options)
ax_thetaf1.axvline(x=true_params.thetaf_1, linestyle="--", color="black")

ax_thetaf2.set_title(r"$E_f^{(2)}$")
ax_thetaf2.hist(nuts.thetaf_2.flatten(), **options)
ax_thetaf2.hist(rwmh.thetaf_2.flatten(), **options)
ax_thetaf2.axvline(x=true_params.thetaf_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(nuts.K_het.flatten(), **options)
ax_Khet.hist(rwmh.K_het.flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)

plt.show()

# %% ESS

num_points = 20

nuts_chain_len = nuts.alpha_1.shape[1]
rwmh_chain_len = rwmh.alpha_1.shape[1]

nuts_end_idx = jnp.linspace(
    nuts_chain_len / num_points, nuts_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

nuts_ess = np.zeros(shape=(7, num_points))
rwmh_ess = np.zeros(shape=(7, num_points))

ess_single: Callable = jit(blackjax.diagnostics.effective_sample_size)

for i, (h_ei, r_ei) in enumerate(zip(nuts_end_idx, rwmh_end_idx)):
    nuts_ess[0, i] = ess_single(nuts.alpha_1[:, :h_ei])
    nuts_ess[1, i] = ess_single(nuts.K0_1[:, :h_ei])
    nuts_ess[2, i] = ess_single(nuts.thetaf_1[:, :h_ei])
    nuts_ess[3, i] = ess_single(nuts.alpha_2[:, :h_ei])
    nuts_ess[4, i] = ess_single(nuts.K0_2[:, :h_ei])
    nuts_ess[5, i] = ess_single(nuts.thetaf_2[:, :h_ei])
    nuts_ess[6, i] = ess_single(nuts.K_het[:, :h_ei])

    rwmh_ess[0, i] = ess_single(rwmh.alpha_1[:, :r_ei])
    rwmh_ess[1, i] = ess_single(rwmh.K0_1[:, :r_ei])
    rwmh_ess[2, i] = ess_single(rwmh.thetaf_1[:, :r_ei])
    rwmh_ess[3, i] = ess_single(rwmh.alpha_2[:, :r_ei])
    rwmh_ess[4, i] = ess_single(rwmh.K0_2[:, :r_ei])
    rwmh_ess[5, i] = ess_single(rwmh.thetaf_2[:, :r_ei])
    rwmh_ess[6, i] = ess_single(rwmh.K_het[:, :r_ei])

# %%  Plot ESS

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.set_ylabel("ESS")

ax_K1 = fig.add_subplot(gs[0, 1])
ax_K1.set_title(r"$K_0^{(1)}$")

ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_thetaf1.set_title(r"$E_f^{(1)}$")

ax_a2 = fig.add_subplot(gs[1, 0])
ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.set_xlabel("Sample Proportion")
ax_a2.set_ylabel("ESS")

ax_K2 = fig.add_subplot(gs[1, 1])
ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.set_xlabel("Sample Proportion")

ax_thetaf2 = fig.add_subplot(gs[1, 2])
ax_thetaf2.set_title(r"$E_f^{(2)}$")
ax_thetaf2.set_xlabel("Sample Proportion")

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])
ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.set_xlabel("Sample Proportion")

axs = [ax_a1, ax_K1, ax_thetaf1, ax_a2, ax_K2, ax_thetaf2, ax_Khet]

idx = jnp.linspace(1 / num_points, 1, num_points)

for i, ax in enumerate(axs):
    ax.plot(idx, nuts_ess[i, :], label="NUTS")
    ax.plot(idx, rwmh_ess[i, :], label="RWMH")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=2)
plt.show()


# %% Sampling: Seed 42

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.02,seed=42.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

nuts: HeterogenousReactionParams = data["nuts"]
rwmh: HeterogenousReactionParams = data["rwmh"]

true_params: HeterogenousReactionParams = HeterogeneousReaction().true_parameters

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_K1 = fig.add_subplot(gs[0, 1])
ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_a2 = fig.add_subplot(gs[1, 0])
ax_K2 = fig.add_subplot(gs[1, 1])
ax_thetaf2 = fig.add_subplot(gs[1, 2])

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])

options = {"density": True, "bins": 50, "alpha": 0.8, "histtype": "step"}

ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.hist(nuts.alpha_1.flatten(), label="NUTS", **options)
ax_a1.hist(rwmh.alpha_1.flatten(), **options, label="RWMH")
ax_a1.axvline(x=true_params.alpha_1, linestyle="--", color="black", label="True Value")

ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.hist(nuts.alpha_2.flatten(), **options)
ax_a2.hist(rwmh.alpha_2.flatten(), **options)
ax_a2.axvline(x=true_params.alpha_2, linestyle="--", color="black")

ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.hist(nuts.K0_1.flatten(), **options)
ax_K1.hist(rwmh.K0_1.flatten(), **options)
ax_K1.axvline(x=true_params.K0_1, linestyle="--", color="black")

ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.hist(nuts.K0_2.flatten(), **options)
ax_K2.hist(rwmh.K0_2.flatten(), **options)
ax_K2.axvline(x=true_params.K0_2, linestyle="--", color="black")

ax_thetaf1.set_title(r"$E_f^{(1)}$")
ax_thetaf1.hist(nuts.thetaf_1.flatten(), **options)
ax_thetaf1.hist(rwmh.thetaf_1.flatten(), **options)
ax_thetaf1.axvline(x=true_params.thetaf_1, linestyle="--", color="black")

ax_thetaf2.set_title(r"$E_f^{(2)}$")
ax_thetaf2.hist(nuts.thetaf_2.flatten(), **options)
ax_thetaf2.hist(rwmh.thetaf_2.flatten(), **options)
ax_thetaf2.axvline(x=true_params.thetaf_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(nuts.K_het.flatten(), **options)
ax_Khet.hist(rwmh.K_het.flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)

plt.show()

# %% ESS

num_points = 20

nuts_chain_len = nuts.alpha_1.shape[1]
rwmh_chain_len = rwmh.alpha_1.shape[1]

nuts_end_idx = jnp.linspace(
    nuts_chain_len / num_points, nuts_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

nuts_ess = np.zeros(shape=(7, num_points))
rwmh_ess = np.zeros(shape=(7, num_points))

ess_single: Callable = jit(blackjax.diagnostics.effective_sample_size)

for i, (h_ei, r_ei) in enumerate(zip(nuts_end_idx, rwmh_end_idx)):
    nuts_ess[0, i] = ess_single(nuts.alpha_1[:, :h_ei])
    nuts_ess[1, i] = ess_single(nuts.K0_1[:, :h_ei])
    nuts_ess[2, i] = ess_single(nuts.thetaf_1[:, :h_ei])
    nuts_ess[3, i] = ess_single(nuts.alpha_2[:, :h_ei])
    nuts_ess[4, i] = ess_single(nuts.K0_2[:, :h_ei])
    nuts_ess[5, i] = ess_single(nuts.thetaf_2[:, :h_ei])
    nuts_ess[6, i] = ess_single(nuts.K_het[:, :h_ei])

    rwmh_ess[0, i] = ess_single(rwmh.alpha_1[:, :r_ei])
    rwmh_ess[1, i] = ess_single(rwmh.K0_1[:, :r_ei])
    rwmh_ess[2, i] = ess_single(rwmh.thetaf_1[:, :r_ei])
    rwmh_ess[3, i] = ess_single(rwmh.alpha_2[:, :r_ei])
    rwmh_ess[4, i] = ess_single(rwmh.K0_2[:, :r_ei])
    rwmh_ess[5, i] = ess_single(rwmh.thetaf_2[:, :r_ei])
    rwmh_ess[6, i] = ess_single(rwmh.K_het[:, :r_ei])

# %%  Plot ESS

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.set_ylabel("ESS")

ax_K1 = fig.add_subplot(gs[0, 1])
ax_K1.set_title(r"$K_0^{(1)}$")

ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_thetaf1.set_title(r"$E_f^{(1)}$")

ax_a2 = fig.add_subplot(gs[1, 0])
ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.set_xlabel("Sample Proportion")
ax_a2.set_ylabel("ESS")

ax_K2 = fig.add_subplot(gs[1, 1])
ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.set_xlabel("Sample Proportion")

ax_thetaf2 = fig.add_subplot(gs[1, 2])
ax_thetaf2.set_title(r"$E_f^{(2)}$")
ax_thetaf2.set_xlabel("Sample Proportion")

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])
ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.set_xlabel("Sample Proportion")

axs = [ax_a1, ax_K1, ax_thetaf1, ax_a2, ax_K2, ax_thetaf2, ax_Khet]

idx = jnp.linspace(1 / num_points, 1, num_points)

for i, ax in enumerate(axs):
    ax.plot(idx, nuts_ess[i, :], label="NUTS")
    ax.plot(idx, rwmh_ess[i, :], label="RWMH")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=2)
plt.show()
