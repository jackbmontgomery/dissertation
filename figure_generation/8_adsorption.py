import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import grad, jit, vmap
from matplotlib.lines import Line2D

from src.diagnostics import potential_scale_reduction_over_time
from src.fdm import (
    AdsorptionReactionBackwardImplicitFDSolver,
    AdsorptionReactionExplicitFDSolver,
    AdsorptionReactionNewtonFDSolver,
)
from src.params import AdsorptionReactionParams, param_property_names
from src.reaction import AdsorptionReaction
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2)

# %%

voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=10)

fd_solver = AdsorptionReactionNewtonFDSolver(voltammetry)
params = AdsorptionReaction().true_parameters

current = fd_solver.solve(params)

plt.plot(fd_solver.applied_potentials, current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

# %% Analytical Flux

cyclic_dc = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=10)
fd_solver = AdsorptionReactionNewtonFDSolver(cyclic_dc)

mono_layer_params = AdsorptionReactionParams(
    alpha_sol=jnp.array(0.5),
    K0_sol=jnp.array(0.0),
    thetaf_sol=jnp.array(0.0),
    alpha_ads=jnp.array(0.5),
    K0_ads=jnp.array(1e6),
    K_A_ads=jnp.array(10e3),
    K_A_des=jnp.array(1e-3),
    K_B_ads=jnp.array(1.0),
    K_B_des=jnp.array(1e-3),
)


mono_current = fd_solver.solve(mono_layer_params)

mono_analytical_flux = (
    -cyclic_dc.sigma
    * jnp.exp(-(fd_solver.applied_potentials - mono_layer_params.thetaf_ads))
    / (
        (1.0 + jnp.exp(-(fd_solver.applied_potentials - mono_layer_params.thetaf_ads)))
        ** 2
    )
)


plt.plot(
    fd_solver.applied_potentials,
    mono_analytical_flux,
    c="C3",
    linestyle=":",
    linewidth=2.5,
    label="Analytical",
)

plt.plot(
    fd_solver.applied_potentials,
    -1 * mono_analytical_flux,
    c="C3",
    linestyle=":",
    linewidth=2.5,
)

plt.plot(fd_solver.applied_potentials, mono_current, label="Numerical")


plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()

plt.savefig("./manuscript/figures/8-analytical.png", dpi=1000)
plt.show()

# %% Numerics Comparison

cyclic_dc = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=10)

newton_solver = AdsorptionReactionNewtonFDSolver(cyclic_dc, atol=1e-12, rtol=1e-10)

T = jnp.linspace(
    cyclic_dc.t_min,
    cyclic_dc.t_max,
    int((cyclic_dc.t_max - cyclic_dc.t_min) / newton_solver.dt),
)

explicit_solver = AdsorptionReactionExplicitFDSolver(cyclic_dc)
backward_solver = AdsorptionReactionBackwardImplicitFDSolver(cyclic_dc)

params = AdsorptionReaction().true_parameters

newton_current = newton_solver.solve(params)
explicit_current = explicit_solver.solve(params)
backward_current = backward_solver.solve(params)

exp_diff = jnp.abs(explicit_current - newton_current) / jnp.abs(newton_current)
bwd_diff = jnp.abs(backward_current - newton_current) / jnp.abs(newton_current)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

ax1.plot(T, newton_current, c="C0", label="Newton Reference")
ax2.plot(T, exp_diff, c="C1", label="Explicit")
ax2.plot(T, bwd_diff, c="C2", label="Backward Implicit")

ax1.set_ylabel("$J$")
ax2.set_yscale("log")
ax2.set_xlabel("$T$")
ax2.set_ylabel("Error")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

handles = h1 + h2
labels = l1 + l2

fig.legend(handles, labels, loc="lower center", ncol=3)

plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
plt.savefig("./manuscript/figures/8-solver-comparison.png", dpi=1000)
plt.show()

# %% Benchmarking the solvers

exp_grad = grad(lambda x: jnp.sum(explicit_solver.solve(x)))
exp_grad(params)

bwd_grad = grad(lambda x: jnp.sum(backward_solver.solve(x)))
bwd_grad(params)

explicit_solver.solve(params)
exp_grad(params)
backward_solver.solve(params)
bwd_grad(params)

# %% Sampling

dir = "./data/sampling"
file = "reaction=AdsorptionReaction,noise=0.02,seed=0.pkl.gz"
with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)
nuts: AdsorptionReactionParams = data["nuts"]
rwmh: AdsorptionReactionParams = data["rwmh"]
true_params: AdsorptionReactionParams = AdsorptionReaction().true_parameters
params = {
    r"$\alpha^{\mathrm{sol}}$": ("alpha_sol", true_params.alpha_sol),
    r"$K_{0}^{\mathrm{sol}}$": ("K0_sol", true_params.K0_sol),
    r"$\alpha^{\mathrm{ads}}$": ("alpha_ads", true_params.alpha_ads),
    r"$K_{0}^{\mathrm{ads}}$": ("K0_ads", true_params.K0_ads),
    r"$K_{A}^{\mathrm{ads}}$": ("K_A_ads", true_params.K_A_ads),
    r"$K_{A}^{\mathrm{des}}$": ("K_A_des", true_params.K_A_des),
    r"$K_{B}^{\mathrm{ads}}$": ("K_B_ads", true_params.K_B_ads),
    r"$K_{B}^{\mathrm{des}}$": ("K_B_des", true_params.K_B_des),
}


def make_df(samples, sampler_name):
    d = {}
    for label, (attr, _) in params.items():
        d[label] = getattr(samples, attr).flatten()
    d["Sampler"] = sampler_name
    return pd.DataFrame(d)


df = pd.concat([make_df(nuts, "NUTS"), make_df(rwmh, "RWMH")], ignore_index=True)
vars_list = list(params.keys())
df_nuts = df[df["Sampler"] == "NUTS"]
g = sns.PairGrid(
    df,
    vars=vars_list,
    hue="Sampler",
    palette={"NUTS": "C0", "RWMH": "C1"},
    corner=True,
    diag_sharey=False,
)
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
            thresh=0.05,
            fill=False,
            linewidths=1.2,
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
g.figure.set_size_inches(14, 14)
plt.tight_layout()
problem_labels = [
    r"$K_{0}^{\mathrm{sol}}$",
    r"$K_{0}^{\mathrm{ads}}$",
    r"$\alpha^{\mathrm{ads}}$",
    r"$\alpha^{\mathrm{sol}}$",
    r"$K_{B}^{\mathrm{ads}}$",
    r"$K_{B}^{\mathrm{des}}$",
]
for idx, label in enumerate(vars_list):
    if label in problem_labels:
        for row in range(idx, len(vars_list)):
            ax = g.axes[row, idx]
            if ax is not None:
                ax.xaxis.set_tick_params(labelsize=6)
plt.savefig("./manuscript/figures/8-corner.png", dpi=1000)
plt.show()


# %% Current Fit

cyclic_dc = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=10)
fd_solver = AdsorptionReactionNewtonFDSolver(cyclic_dc)
key = jr.key(0)
key_data, key_samples = jr.split(key)

base_current = fd_solver.solve(AdsorptionReaction().true_parameters)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_data,
)

num_samples = 200

sample_indexes = jr.choice(
    key_samples, len(nuts.alpha_sol.flatten()), shape=(num_samples,), replace=False
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
plt.savefig("./manuscript/figures/8-current-fit.png", dpi=1000)
plt.show()

# %% GR over sampling

fig, axs = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
num_points = 50
x = jnp.linspace(1 / num_points, 1, num_points)
param_names = param_property_names(nuts)


params_titles = {
    "alpha_sol": r"$\alpha^{\mathrm{sol}}$",
    "K0_sol": r"$K_{0}^{\mathrm{sol}}$",
    "thetaf_sol": r"$\theta_{f}^{\mathrm{sol}}$",
    "alpha_ads": r"$\alpha^{\mathrm{ads}}$",
    "K0_ads": r"$K_{0}^{\mathrm{ads}}$",
    "K_A_ads": r"$K_{A}^{\mathrm{ads}}$",
    "K_A_des": r"$K_{A}^{\mathrm{des}}$",
    "K_B_ads": r"$K_{B}^{\mathrm{ads}}$",
    "K_B_des": r"$K_{B}^{\mathrm{des}}$",
}

for name, ax in zip(param_names, axs.flatten()):
    ax.set_title(params_titles[name])
    nuts_ess = potential_scale_reduction_over_time(
        getattr(nuts, name), num_points=num_points
    )
    ax.plot(x, nuts_ess, label="NUTS")
    rwmh_ess = potential_scale_reduction_over_time(
        getattr(rwmh, name), num_points=num_points
    )
    ax.plot(x, rwmh_ess, label="RWMH")
    ax.axhline(1.01, color="k", ls="--", lw=0.8, label=r"$1.01$")

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("./manuscript/figures/8-gr.png", dpi=1000)
plt.show()

# %% ESS Table

ess: Callable = jit(blackjax.diagnostics.effective_sample_size)

params = [
    "alpha_sol",
    "K0_sol",
    "thetaf_sol",
    "alpha_ads",
    "K0_ads",
    "K_A_ads",
    "K_A_des",
    "K_B_ads",
    "K_B_des",
]

for name, data in [("NUTS", nuts), ("RWMH", rwmh)]:
    vals = " & ".join(f"{ess(getattr(data, p)):.1f}" for p in params)
    print(f"    {name} & {vals} \\\\")

vals = " & ".join(
    f"{ess(getattr(nuts, p)) / ess(getattr(rwmh, p)):.1f}" for p in params
)
print(f"    Ratio & {vals} \\\\")

# %% Discretisation refinement study

voltammetry = CyclicDC()
params = AdsorptionReaction().true_parameters

# Reference: fine in both space and time

ref_dtheta = 1e-5
ref_h0 = 1e-10
ref_solver = AdsorptionReactionNewtonFDSolver(voltammetry, h0=ref_h0, dtheta=ref_dtheta)
ref_current = ref_solver.solve(params).block_until_ready()
ref_time = jnp.arange(len(ref_current)) * ref_dtheta


def rel_l2_vs_ref(h0, dtheta):
    solver = AdsorptionReactionNewtonFDSolver(voltammetry, h0=h0, dtheta=dtheta)
    current = solver.solve(params).block_until_ready()
    t = jnp.arange(len(current)) * dtheta
    ref_interp = jnp.interp(t, ref_time, ref_current)
    return float(jnp.linalg.norm(current - ref_interp) / jnp.linalg.norm(ref_interp))


# --- Spatial sweep: vary h0 at several dtheta ---
h0_range = jnp.power(10.0, jnp.arange(-9, -2))
dtheta_sweep = [1e-3, 5e-3, 1e-2, 5e-2]
spatial_errors = {dt: [rel_l2_vs_ref(h0, dt) for h0 in h0_range] for dt in dtheta_sweep}

# --- Temporal sweep: vary dtheta at fixed small h0 ---
h0_fixed = 1e-6
dtheta_range = jnp.array([2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 5e-3, 1e-2, 2e-2, 5e-2])
temporal_errors = jnp.array([rel_l2_vs_ref(h0_fixed, dt) for dt in dtheta_range])

# Empirical convergence order (least-squares slope in log-log)

slope, intercept = jnp.polyfit(jnp.log(dtheta_range), jnp.log(temporal_errors), 1)
print(f"Empirical temporal order: {slope:.2f}")

# --- Plot ---
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

plt.savefig("./manuscript/figures/8-discretisation.png", dpi=1000)
plt.show()
