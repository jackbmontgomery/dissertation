import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
plt.axhline(y=max_current, c="C3", linestyle="--", label="Analytical")
plt.axhline(y=2 * max_current, c="C3", linestyle="--", label="Analytical")

max_current_position = (
    jnp.log(params_1.K0_1 / jnp.sqrt(params_1.alpha_1 * voltammetry.sigma)) - 0.78
) / params_1.alpha_1
plt.axvline(x=max_current_position, c="C3", linestyle="--")

current_1 = fd_solver.solve(params_1)
current_2 = fd_solver.solve(params_2)

plt.xlabel(r"$\theta$")
plt.ylabel(r"$J$")
plt.plot(fd_solver.applied_potentials, current_1)
plt.plot(fd_solver.applied_potentials, current_2)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.savefig("./manuscript/figures/7-analytical.png", dpi=1000)
plt.show()

# %% Sampling: Seed 0 Corner plot

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.02,seed=0.pkl.gz"
with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)
nuts: HeterogenousReactionParams = data["nuts"]
rwmh: HeterogenousReactionParams = data["rwmh"]
true_params: HeterogenousReactionParams = HeterogeneousReaction().true_parameters

# Build a combined DataFrame
params = {
    r"$\alpha^{(1)}$": ("alpha_1", true_params.alpha_1),
    r"$\alpha^{(2)}$": ("alpha_2", true_params.alpha_2),
    r"$K_0^{(1)}$": ("K0_1", true_params.K0_1),
    r"$K_0^{(2)}$": ("K0_2", true_params.K0_2),
    r"$K_{\mathrm{het}}$": ("K_het", true_params.K_het),
}

rows = []
for label, (attr, _) in params.items():
    pass  # just defining the mapping


def make_df(samples, sampler_name):
    d = {}
    for label, (attr, _) in params.items():
        d[label] = getattr(samples, attr).flatten()
    d["Sampler"] = sampler_name
    return pd.DataFrame(d)


df = pd.concat([make_df(nuts, "NUTS"), make_df(rwmh, "RWMH")], ignore_index=True)

vars_list = list(params.keys())

g = sns.PairGrid(
    df,
    vars=vars_list,
    hue="Sampler",
    palette={"NUTS": "C0", "RWMH": "C1"},
    corner=True,
    diag_sharey=False,
)

# Diagonal: histograms
g.map_diag(
    sns.histplot,
    stat="density",
    bins=50,
    alpha=0.8,
    element="step",
    fill=False,
    linewidth=1.2,
    common_norm=False,
)

# Lower triangle: KDE
g.map_lower(sns.kdeplot, levels=5, fill=False, linewidths=1.2, thresh=0.05)

# Add true value lines on diagonal
for i, label in enumerate(vars_list):
    ax = g.axes[i, i]
    _, true_val = params[label]
    ax.axvline(x=true_val, linestyle="--", color="black", linewidth=1.0)

# Add true value crosshairs on lower off-diagonal
for i in range(len(vars_list)):
    for j in range(i):
        ax = g.axes[i, j]
        _, true_y = params[vars_list[i]]
        _, true_x = params[vars_list[j]]
        ax.axvline(x=true_x, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(y=true_y, linestyle="--", color="black", linewidth=0.5, alpha=0.5)

g.add_legend()
g.figure.set_size_inches(10, 10)
plt.tight_layout()
plt.savefig("./manuscript/figures/7-corner.png", dpi=1000)
plt.show()

# %% Current Fit

cyclic_dc = CyclicDC()
fd_solver = HeterogeneousReactionFDSolver(cyclic_dc)

true_current = fd_solver.solve(true_params)
nuts_mean = jtu.tree_map(lambda x: jnp.mean(x), nuts)
nuts_current = fd_solver.solve(nuts_mean)

plt.plot(fd_solver.applied_potentials, nuts_current, label="NUTS")
plt.plot(
    fd_solver.applied_potentials,
    true_current,
    c="C3",
    linestyle="--",
    label="True Current",
)

plt.savefig("./manuscript/figures/7-current-fit.png", dpi=1000)
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

fig = plt.figure(figsize=(15, 6))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.set_ylabel("ESS")

ax_K1 = fig.add_subplot(gs[0, 1])
ax_K1.set_title(r"$K_0^{(1)}$")

ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_thetaf1.set_title(r"$\theta_f^{(1)}$")

ax_a2 = fig.add_subplot(gs[1, 0])
ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.set_xlabel("Sample Proportion")
ax_a2.set_ylabel("ESS")

ax_K2 = fig.add_subplot(gs[1, 1])
ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.set_xlabel("Sample Proportion")

ax_thetaf2 = fig.add_subplot(gs[1, 2])
ax_thetaf2.set_title(r"$\theta_f^{(2)}$")
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
fig.legend(handles, labels, loc="lower right", ncol=1)
plt.savefig("./manuscript/figures/7-ess.png", dpi=1000)
plt.show()


# %%

ac_voltammetry = CyclicAC()
dc_voltammetry = CyclicDC()

fd_solver = HeterogeneousReactionFDSolver(ac_voltammetry)
linear_applied_potentials = HeterogeneousReactionFDSolver(
    dc_voltammetry
).applied_potentials
params = HeterogeneousReaction().true_parameters
current = fd_solver.solve(params)

plt.plot(linear_applied_potentials, current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.savefig("./manuscript/figures/7-ac-voltam.png", dpi=1000)
plt.show()

# %% Sampling: AC vs DC

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.02,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    dc_data = pickle.load(f)

file = "reaction=HeterogeneousReaction,noise=0.02,seed=1.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    ac_data = pickle.load(f)

dc_nuts: HeterogenousReactionParams = dc_data["nuts"]
ac_nuts: HeterogenousReactionParams = ac_data["nuts"]

true_params: HeterogenousReactionParams = HeterogeneousReaction().true_parameters

fig = plt.figure(figsize=(15, 6))

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
ax_a1.hist(dc_nuts.alpha_1.flatten(), label="DC", **options)
ax_a1.hist(ac_nuts.alpha_1.flatten(), **options, label="AC")
ax_a1.axvline(x=true_params.alpha_1, linestyle="--", color="black", label="True Value")

ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.hist(dc_nuts.alpha_2.flatten(), **options)
ax_a2.hist(ac_nuts.alpha_2.flatten(), **options)
ax_a2.axvline(x=true_params.alpha_2, linestyle="--", color="black")

ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.hist(dc_nuts.K0_1.flatten(), **options)
ax_K1.hist(ac_nuts.K0_1.flatten(), **options)
ax_K1.axvline(x=true_params.K0_1, linestyle="--", color="black")

ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.hist(dc_nuts.K0_2.flatten(), **options)
ax_K2.hist(ac_nuts.K0_2.flatten(), **options)
ax_K2.axvline(x=true_params.K0_2, linestyle="--", color="black")

ax_thetaf1.set_title(r"$\theta_f^{(1)}$")
ax_thetaf1.hist(dc_nuts.thetaf_1.flatten(), **options)
ax_thetaf1.hist(ac_nuts.thetaf_1.flatten(), **options)
ax_thetaf1.axvline(x=true_params.thetaf_1, linestyle="--", color="black")

ax_thetaf2.set_title(r"$\theta_f^{(2)}$")
ax_thetaf2.hist(dc_nuts.thetaf_2.flatten(), **options)
ax_thetaf2.hist(ac_nuts.thetaf_2.flatten(), **options)
ax_thetaf2.axvline(x=true_params.thetaf_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(dc_nuts.K_het.flatten(), **options)
ax_Khet.hist(ac_nuts.K_het.flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)
plt.savefig("./manuscript/figures/7-ac-hist-comparison.png", dpi=1000)
plt.show()

# %% Numerical Convergence Checks

voltammetry = CyclicDC()
params = HeterogeneousReaction().true_parameters
base_dtheta = 1e-4
base_h0 = 1e-10
fd_solver = HeterogeneousReactionFDSolver(voltammetry, h0=base_h0, dtheta=base_dtheta)
base_current = fd_solver.solve(params).block_until_ready()
base_time = jnp.arange(len(base_current)) * base_dtheta

h0_range = jnp.power(10.0, jnp.arange(-9, -2))
dtheta_range = [2e-4, 5e-4, 8e-4, 1e-3, 2e-3, 5e-3, 1e-2]

for dtheta in dtheta_range:
    dtheta_vals = []
    coarse_time = jnp.arange(int(base_time[-1] / dtheta) + 1) * dtheta
    base_interp = jnp.interp(coarse_time, base_time, base_current)
    for h0 in h0_range:
        fd_solver = HeterogeneousReactionFDSolver(voltammetry, h0=h0, dtheta=dtheta)
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

ref_dtheta = 1e-5
ref_h0 = 1e-10
ref_solver = HeterogeneousReactionFDSolver(voltammetry, h0=ref_h0, dtheta=ref_dtheta)
ref_current = ref_solver.solve(params).block_until_ready()
ref_time = jnp.arange(len(ref_current)) * ref_dtheta

# Fix h0 in the flat region, sweep dtheta
h0_fixed = 1e-8
dtheta_range = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
errors = []

for dtheta in dtheta_range:
    solver = HeterogeneousReactionFDSolver(voltammetry, h0=h0_fixed, dtheta=dtheta)
    current = solver.solve(params).block_until_ready()
    coarse_time = jnp.arange(len(current)) * dtheta
    ref_interp = jnp.interp(coarse_time, ref_time, ref_current)
    rel_l2 = float(jnp.linalg.norm(current - ref_interp) / jnp.linalg.norm(ref_interp))
    errors.append(rel_l2)
    print(f"dtheta = {dtheta:.4f}, rel L2 error = {rel_l2:.2e}")

print("\nConvergence order:")
for i in range(1, len(dtheta_range)):
    ratio = errors[i] / errors[i - 1]
    dt_ratio = dtheta_range[i] / dtheta_range[i - 1]
    order = jnp.log(ratio) / jnp.log(dt_ratio)
    print(
        f"dtheta {dtheta_range[i - 1]:.4f} -> {dtheta_range[i]:.4f}: order = {float(order):.2f}"
    )
