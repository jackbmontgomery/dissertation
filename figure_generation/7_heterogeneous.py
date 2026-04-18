import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import jit, vmap
from matplotlib.lines import Line2D

from src.fdm import HeterogeneousReactionFDSolver
from src.params import HeterogenousReactionParams
from src.reaction import HeterogeneousReaction
from src.utils import generate_noisy_samples
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
plt.axhline(y=max_current, c="black", linestyle="--", label="Analytical")
plt.axhline(y=2 * max_current, c="black", linestyle="--")

max_current_position = (
    jnp.log(params_1.K0_1 / jnp.sqrt(params_1.alpha_1 * voltammetry.sigma)) - 0.78
) / params_1.alpha_1
plt.axvline(x=max_current_position, c="black", linestyle="--")

current_1 = fd_solver.solve(params_1)
current_2 = fd_solver.solve(params_2)

plt.xlabel(r"$\theta$")
plt.ylabel(r"$J$")
plt.plot(fd_solver.applied_potentials, current_1, label="One-electron")
plt.plot(fd_solver.applied_potentials, current_2, label="Two-electron")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.legend()
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

# Filter to NUTS-only for KDE
df_nuts = df[df["Sampler"] == "NUTS"]

g = sns.PairGrid(
    df,
    vars=vars_list,
    hue="Sampler",
    palette={"NUTS": "C0", "RWMH": "C1"},
    corner=True,
    diag_sharey=False,
)

# Diagonal: both samplers
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

# Lower triangle: NUTS only
for i in range(len(vars_list)):
    for j in range(i):
        ax = g.axes[i, j]
        sns.kdeplot(
            x=df_nuts[vars_list[j]],
            y=df_nuts[vars_list[i]],
            ax=ax,
            levels=5,
            fill=False,
            linewidths=1.2,
            thresh=0.05,
            color="C0",
        )

# True value lines on diagonal
for i, label in enumerate(vars_list):
    ax = g.axes[i, i]
    _, true_val = params[label]
    ax.axvline(x=true_val, linestyle="--", color="black", linewidth=1.0)

# True value crosshairs on lower off-diagonal
for i in range(len(vars_list)):
    for j in range(i):
        ax = g.axes[i, j]
        _, true_y = params[vars_list[i]]
        _, true_x = params[vars_list[j]]
        ax.axvline(x=true_x, linestyle="--", color="black", linewidth=1.0, alpha=0.5)
        ax.axhline(y=true_y, linestyle="--", color="black", linewidth=1.0, alpha=0.5)

handles = [
    Line2D([0], [0], color="C0", linewidth=1.2, label="NUTS"),
    Line2D([0], [0], color="C1", linewidth=1.2, label="RWMH"),
]
g.figure.legend(handles=handles, title="Sampler", loc="upper right", frameon=True)
g.figure.set_size_inches(10, 10)
plt.tight_layout()
plt.savefig("./manuscript/figures/7-corner.png", dpi=1000)
plt.show()

# % Current Fit

cyclic_dc = CyclicDC()
fd_solver = HeterogeneousReactionFDSolver(cyclic_dc)
key = jr.key(0)
key_data, key_samples = jr.split(key)

base_current = fd_solver.solve(HeterogeneousReaction().true_parameters)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_data,
)

num_samples = 200

sample_indexes = jr.choice(
    key_samples, len(nuts.alpha_1.flatten()), shape=(num_samples,), replace=False
)

nuts_samples = jtu.tree_map(lambda x: x.flatten()[sample_indexes], nuts)

currents = vmap(fd_solver.solve)(nuts_samples)

mean_current = jnp.mean(currents, axis=0)
lower = jnp.percentile(currents, 2.5, axis=0)
upper = jnp.percentile(currents, 97.5, axis=0)


plt.figure(figsize=(10, 8))

for i, samples in enumerate(experimental_samples):
    if i == 0:
        label = "Noisy Data"
    else:
        label = None

    plt.scatter(
        fd_solver.applied_potentials,
        samples,
        s=5,
        c="C3",
        alpha=0.5,
        label=label,
    )


plt.plot(
    fd_solver.applied_potentials,
    base_current,
    linestyle="--",
    c="black",
    linewidth=2.0,
    label="True Current",
)

plt.fill_between(
    fd_solver.applied_potentials,
    lower,
    upper,
    alpha=0.5,
    label="95% credible interval",
)
plt.plot(fd_solver.applied_potentials, mean_current, label="Posterior mean")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.ylabel("$J$")
plt.xlabel(r"$\theta$")

plt.legend(markerscale=5)
plt.savefig("./manuscript/figures/7-current-fit.png", dpi=1000)
plt.show()

# %% GR

num_points = 50

nuts_chain_len = nuts.alpha_1.shape[1]
rwmh_chain_len = rwmh.alpha_1.shape[1]

nuts_end_idx = jnp.linspace(
    nuts_chain_len / num_points, nuts_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

nuts_gr = np.zeros(shape=(7, num_points))
rwmh_gr = np.zeros(shape=(7, num_points))

gr_single: Callable = jit(blackjax.diagnostics.potential_scale_reduction)

for i, (h_ei, r_ei) in enumerate(zip(nuts_end_idx, rwmh_end_idx)):
    nuts_gr[0, i] = gr_single(nuts.alpha_1[:, :h_ei])
    nuts_gr[1, i] = gr_single(nuts.K0_1[:, :h_ei])
    nuts_gr[2, i] = gr_single(nuts.thetaf_1[:, :h_ei])
    nuts_gr[3, i] = gr_single(nuts.alpha_2[:, :h_ei])
    nuts_gr[4, i] = gr_single(nuts.K0_2[:, :h_ei])
    nuts_gr[5, i] = gr_single(nuts.thetaf_2[:, :h_ei])
    nuts_gr[6, i] = gr_single(nuts.K_het[:, :h_ei])

    rwmh_gr[0, i] = gr_single(rwmh.alpha_1[:, :r_ei])
    rwmh_gr[1, i] = gr_single(rwmh.K0_1[:, :r_ei])
    rwmh_gr[2, i] = gr_single(rwmh.thetaf_1[:, :r_ei])
    rwmh_gr[3, i] = gr_single(rwmh.alpha_2[:, :r_ei])
    rwmh_gr[4, i] = gr_single(rwmh.K0_2[:, :r_ei])
    rwmh_gr[5, i] = gr_single(rwmh.thetaf_2[:, :r_ei])
    rwmh_gr[6, i] = gr_single(rwmh.K_het[:, :r_ei])

# %%  Plot GR

fig = plt.figure(figsize=(15, 6))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.set_ylabel(r"$\hat{R}$")
ax_a1.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

ax_K1 = fig.add_subplot(gs[0, 1])
ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

ax_thetaf1 = fig.add_subplot(gs[0, 2])
ax_thetaf1.set_title(r"$\theta_f^{(1)}$")
ax_thetaf1.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

ax_a2 = fig.add_subplot(gs[1, 0])
ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.set_xlabel("Sample Proportion")
ax_a2.set_ylabel(r"$\hat{R}$")
ax_a2.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

ax_K2 = fig.add_subplot(gs[1, 1])
ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.set_xlabel("Sample Proportion")
ax_K2.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

ax_thetaf2 = fig.add_subplot(gs[1, 2])
ax_thetaf2.set_title(r"$\theta_f^{(2)}$")
ax_thetaf2.set_xlabel("Sample Proportion")
ax_thetaf2.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])
ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.set_xlabel("Sample Proportion")
ax_Khet.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

axs = [ax_a1, ax_K1, ax_thetaf1, ax_a2, ax_K2, ax_thetaf2, ax_Khet]

idx = jnp.linspace(1 / num_points, 1, num_points)

for i, ax in enumerate(axs):
    ax.plot(idx, nuts_gr[i, :], label="NUTS")
    ax.plot(idx, rwmh_gr[i, :], label="RWMH")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)
plt.savefig("./manuscript/figures/7-gr.png", dpi=1000)
plt.show()

# %% ESS Table

ess: Callable = jit(blackjax.diagnostics.effective_sample_size)

params = [
    "alpha_1",
    "K0_1",
    "thetaf_1",
    "alpha_2",
    "K0_2",
    "thetaf_2",
    "K_het",
]

for name, data in [("NUTS", nuts), ("RWMH", rwmh)]:
    vals = " & ".join(f"{ess(getattr(data, p)):.1f}" for p in params)
    print(f"    {name} & {vals} \\\\")

vals = " & ".join(
    f"{ess(getattr(nuts, p)) / ess(getattr(rwmh, p)):.1f}" for p in params
)
print(f"    Ratio & {vals} \\\\")

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

# %% Discretisation refinement study

voltammetry = CyclicDC()
params = HeterogeneousReaction().true_parameters

ref_dtheta = 1e-5
ref_h0 = 1e-10
ref_solver = HeterogeneousReactionFDSolver(voltammetry, h0=ref_h0, dtheta=ref_dtheta)
ref_current = ref_solver.solve(params).block_until_ready()
ref_time = jnp.arange(len(ref_current)) * ref_dtheta


def rel_l2_vs_ref(h0, dtheta):
    solver = HeterogeneousReactionFDSolver(voltammetry, h0=h0, dtheta=dtheta)
    current = solver.solve(params).block_until_ready()
    t = jnp.arange(len(current)) * dtheta
    ref_interp = jnp.interp(t, ref_time, ref_current)
    return float(jnp.linalg.norm(current - ref_interp) / jnp.linalg.norm(ref_interp))


h0_range = jnp.power(10.0, jnp.arange(-9, -2))
dtheta_sweep = [1e-3, 5e-3, 1e-2, 5e-2]
spatial_errors = {dt: [rel_l2_vs_ref(h0, dt) for h0 in h0_range] for dt in dtheta_sweep}

h0_fixed = 1e-6
dtheta_range = jnp.array([2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 1e-2, 2e-2, 5e-2])
temporal_errors = jnp.array([rel_l2_vs_ref(h0_fixed, dt) for dt in dtheta_range])


slope, intercept = jnp.polyfit(jnp.log(dtheta_range), jnp.log(temporal_errors), 1)
print(f"Empirical temporal order: {slope:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
for dt, errs in spatial_errors.items():
    ax.plot(h0_range, errs, marker="o", label=rf"$\Delta\theta = {dt:g}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$h_0$")
ax.set_ylabel(r"Relative $L^2$ error")
ax.axvline(1e-6, color="k", linestyle=":", linewidth=1)
ax.legend(fontsize=12)

ax = axes[1]
ax.plot(dtheta_range, temporal_errors, marker="o", label="Measured")
ref_line = temporal_errors[0] * (dtheta_range / dtheta_range[0]) ** 1.0
ax.plot(dtheta_range, ref_line, linestyle="--", label=r"Slope 1")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Delta\theta$")
ax.axvline(4e-3, color="k", linestyle=":", linewidth=1)
ax.legend(fontsize=12)

fig.tight_layout()

plt.savefig("./manuscript/figures/7-discretisation.png", dpi=1000)
plt.show()
