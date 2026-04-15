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
from jax import jit, vmap
from matplotlib.lines import Line2D

from src.diagnostics import ess_over_time
from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams, param_property_names
from src.reaction import ElectronReaction, ReversibleElectronReaction
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2.0)

# %% Quasi-reverisble

dir = "./data/sampling/"
file = "reaction=ElectronReaction,noise=0.02,seed=0.pkl.gz"
with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)
nuts_quasi: ElectronReactionParams = data["nuts"]
rwmh_quasi: ElectronReactionParams = data["rwmh"]
true_params_quasi = ElectronReaction().true_parameters

params = {
    r"$\alpha$": ("alpha", true_params_quasi.alpha),
    r"$K_0$": ("K0", true_params_quasi.K0),
    r"$\theta_f$": ("thetaf", true_params_quasi.thetaf),
}


def make_df(samples, sampler_name):
    d = {}
    for label, (attr, _) in params.items():
        d[label] = getattr(samples, attr).flatten()
    d["Sampler"] = sampler_name
    return pd.DataFrame(d)


df = pd.concat(
    [make_df(nuts_quasi, "NUTS"), make_df(rwmh_quasi, "RWMH")], ignore_index=True
)
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

for i, label in enumerate(vars_list):
    ax = g.axes[i, i]
    _, true_val = params[label]
    ax.axvline(x=true_val, linestyle="--", color="black", linewidth=1.0)

for i in range(len(vars_list)):
    for j in range(i):
        ax = g.axes[i, j]
        _, true_y = params[vars_list[i]]
        _, true_x = params[vars_list[j]]
        ax.axvline(x=true_x, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(y=true_y, linestyle="--", color="black", linewidth=0.5, alpha=0.5)

handles = [
    Line2D([0], [0], color="C0", linewidth=1.2, label="NUTS"),
    Line2D([0], [0], color="C1", linewidth=1.2, label="RWMH"),
]

g.figure.legend(handles=handles, title="Sampler", loc="upper right", frameon=True)
g.figure.set_size_inches(8, 8)
plt.tight_layout()
plt.savefig("./manuscript/figures/6-corner-quasi.png", dpi=1000)
plt.show()

# %%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
num_points = 20
x = jnp.linspace(1 / num_points, 1, num_points)
param_names = param_property_names(nuts_quasi)
headers = [r"$\alpha$", r"$K_0$", r"$\theta_f$"]

ax1.set_title(r"$\alpha$")
nuts_ess = ess_over_time(nuts_quasi.alpha, num_points=num_points)
ax1.plot(x, nuts_ess, label="NUTS")
rwmh_ess = ess_over_time(rwmh_quasi.alpha, num_points=num_points)
ax1.plot(x, rwmh_ess, label="RWMH")

ax2.set_title(r"$K_0$")
nuts_ess = ess_over_time(nuts_quasi.K0, num_points=num_points)
ax2.plot(x, nuts_ess, label="NUTS")
rwmh_ess = ess_over_time(rwmh_quasi.K0, num_points=num_points)
ax2.plot(x, rwmh_ess, label="RWMH")

ax3.set_title(r"$\theta_f$")
nuts_ess = ess_over_time(nuts_quasi.thetaf, num_points=num_points)
ax3.plot(x, nuts_ess, label="NUTS")
rwmh_ess = ess_over_time(rwmh_quasi.thetaf, num_points=num_points)
ax3.plot(x, rwmh_ess, label="RWMH")

ax1.set_ylabel("ESS")
ax2.set_xlabel("Sample Proportion")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/6-ess.png", dpi=1000)
plt.show()

# %% ESS

ess: Callable = jit(blackjax.diagnostics.effective_sample_size)

print(
    f"NUTS & {ess(nuts_quasi.alpha):.1f} & {ess(nuts_quasi.K0):.1f} & {ess(nuts_quasi.thetaf):.1f}"
)
print(
    f"RWMH & {ess(rwmh_quasi.alpha):.1f} & {ess(rwmh_quasi.K0):.1f} & {ess(rwmh_quasi.thetaf):.1f}"
)

# %% Reversible

dir = "./data/sampling/"
file = "reaction=ElectronReaction,noise=0.02,seed=1.pkl.gz"
with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)
nuts_rev: ElectronReactionParams = data["nuts"]
rwmh_rev: ElectronReactionParams = data["rwmh"]
true_params_rev = ReversibleElectronReaction().true_parameters

params = {
    r"$\alpha$": ("alpha", true_params_rev.alpha),
    r"$K_0$": ("K0", true_params_rev.K0),
    r"$\theta_f$": ("thetaf", true_params_rev.thetaf),
}


def make_df(samples, sampler_name):
    d = {}
    for label, (attr, _) in params.items():
        d[label] = getattr(samples, attr).flatten()
    d["Sampler"] = sampler_name
    return pd.DataFrame(d)


df = pd.concat(
    [make_df(nuts_rev, "NUTS"), make_df(rwmh_rev, "RWMH")], ignore_index=True
)
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

for i, label in enumerate(vars_list):
    ax = g.axes[i, i]
    _, true_val = params[label]
    ax.axvline(x=true_val, linestyle="--", color="black", linewidth=1.0)

for i in range(len(vars_list)):
    for j in range(i):
        ax = g.axes[i, j]
        _, true_y = params[vars_list[i]]
        _, true_x = params[vars_list[j]]
        ax.axvline(x=true_x, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(y=true_y, linestyle="--", color="black", linewidth=0.5, alpha=0.5)

handles = [
    Line2D([0], [0], color="C0", linewidth=1.2, label="NUTS"),
    Line2D([0], [0], color="C1", linewidth=1.2, label="RWMH"),
]
g.figure.legend(handles=handles, title="Sampler", loc="upper right", frameon=True)
g.figure.set_size_inches(8, 8)
plt.tight_layout()
plt.savefig("./manuscript/figures/6-corner-rev.png", dpi=1000)
plt.show()

# %% Current fits

cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)
key = jr.key(0)
key_data, key_samples = jr.split(key)


num_samples = 200


def gen_fig(ax, nuts, reaction):
    base_current = fd_solver.solve(reaction.true_parameters)

    experimental_samples = generate_noisy_samples(
        10,
        base_current,
        0.02,
        key=key_data,
    )

    sample_indexes = jr.choice(
        key_samples,
        len(nuts.alpha.flatten()),
        shape=(num_samples,),
        replace=False,
    )
    nuts_samples = jtu.tree_map(lambda x: x.flatten()[sample_indexes], nuts)

    currents = vmap(fd_solver.solve)(nuts_samples)

    mean_current = jnp.mean(currents, axis=0)
    lower = jnp.percentile(currents, 2.5, axis=0)
    upper = jnp.percentile(currents, 97.5, axis=0)

    for i, samples in enumerate(experimental_samples):
        if i == 0:
            label = "Noisy Data"
        else:
            label = None

        ax.scatter(
            fd_solver.applied_potentials,
            samples,
            s=5,
            c="C3",
            alpha=0.3,
            label=label,
        )

    ax.plot(
        fd_solver.applied_potentials,
        base_current,
        linestyle="--",
        c="black",
        linewidth=2.0,
        label="True Current",
    )

    ax.fill_between(
        fd_solver.applied_potentials,
        lower,
        upper,
        alpha=0.3,
        label="95% credible interval",
    )
    ax.plot(fd_solver.applied_potentials, mean_current, label="Posterior mean")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
gen_fig(ax1, nuts_quasi, ElectronReaction())
gen_fig(ax2, nuts_rev, ReversibleElectronReaction())

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, markerscale=5)
plt.tight_layout(rect=(0, 0.2, 1, 1))
plt.savefig("./manuscript/figures/6-current-fit.png", dpi=1000)
plt.show()
