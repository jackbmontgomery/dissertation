import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
from jax import jit, vmap

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True, "histtype": "step", "linewidth": 1}

ax1.set_title(r"$\alpha$")
ax1.hist(nuts_quasi.alpha.flatten(), label="NUTS", **options)
ax1.hist(rwmh_quasi.alpha.flatten(), label="RWMH", **options)
ax1.axvline(
    x=true_params_quasi.alpha, linestyle="--", color="black", label="True Value"
)
ax1.set_ylabel("Density")

ax2.set_title(r"$K_0$")
ax2.hist(nuts_quasi.K0.flatten(), **options)
ax2.hist(rwmh_quasi.K0.flatten(), **options)
ax2.axvline(x=true_params_quasi.K0, linestyle="--", color="black")

ax3.set_title(r"$\theta_f$")
ax3.hist(nuts_quasi.thetaf.flatten(), **options)
ax3.hist(rwmh_quasi.thetaf.flatten(), **options)
ax3.axvline(x=true_params_quasi.thetaf, linestyle="--", color="black")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/6-hmc-hist.png", dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
num_points = 20
x = jnp.linspace(1 / num_points, 1, num_points)
param_names = param_property_names(nuts_quasi)
headers = [r"$\alpha$", r"$K_0$", r"$\theta_f$"]

for name, header, ax in zip(param_names, headers, axs):
    ax.set_title(header)
    nuts_ess = ess_over_time(getattr(nuts_quasi, name), num_points=num_points)
    ax.plot(x, nuts_ess, label="NUTS")
    rwmh_ess = ess_over_time(getattr(rwmh_quasi, name), num_points=num_points)
    ax.plot(x, rwmh_ess, label="RWMH")

axs[0].set_ylabel("ESS")
axs[1].set_xlabel("Sample Proportion")

handles, labels = axs[0].get_legend_handles_labels()
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True, "histtype": "step", "linewidth": 1}

ax1.set_title(r"$\alpha$")
ax1.hist(nuts_rev.alpha.flatten(), label="NUTS", **options)
ax1.hist(rwmh_rev.alpha.flatten(), label="RWMH", **options)
ax1.axvline(x=true_params_rev.alpha, linestyle="--", color="black", label="True Value")
ax1.set_ylabel("Density")

ax2.set_title(r"$K_0$")
ax2.hist(nuts_rev.K0.flatten(), **options)
ax2.hist(rwmh_rev.K0.flatten(), **options)
ax2.axvline(x=true_params_rev.K0, linestyle="--", color="black")

ax3.set_title(r"$\theta_f$")
ax3.hist(nuts_rev.thetaf.flatten(), **options)
ax3.hist(rwmh_rev.thetaf.flatten(), **options)
ax3.axvline(x=true_params_rev.thetaf, linestyle="--", color="black")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/6-hmc-hist-rev.png", dpi=1000)
plt.show()


# %% Current fits

cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

true_quasi_current = fd_solver.solve(true_params_quasi)
nuts_quasi_mean = jtu.tree_map(lambda x: jnp.mean(x), nuts_quasi)
nuts_quasi_current = fd_solver.solve(nuts_quasi_mean)
rwmh_quasi_mean = jtu.tree_map(lambda x: jnp.mean(x), rwmh_quasi)
rwmh_quasi_current = fd_solver.solve(rwmh_quasi_mean)

ax1.plot(fd_solver.applied_potentials, nuts_quasi_current)
ax1.plot(fd_solver.applied_potentials, rwmh_quasi_current)
ax1.plot(fd_solver.applied_potentials, true_quasi_current, linestyle="--")

true_rev_current = fd_solver.solve(true_params_rev)
nuts_rev_mean = jtu.tree_map(lambda x: jnp.mean(x), nuts_rev)
nuts_rev_current = fd_solver.solve(nuts_rev_mean)
rwmh_rev_mean = jtu.tree_map(lambda x: jnp.mean(x), rwmh_rev)
rwmh_rev_current = fd_solver.solve(rwmh_rev_mean)

ax2.plot(fd_solver.applied_potentials, nuts_rev_current)
ax2.plot(fd_solver.applied_potentials, rwmh_rev_current)
ax2.plot(fd_solver.applied_potentials, true_rev_current, linestyle="--")

plt.savefig("./manuscript/figures/6-current-fit.png", dpi=1000)
plt.show()

# %% Current fits

cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)
key = jr.key(0)
key_data, key_samples = jr.split(key)

base_current = fd_solver.solve(ElectronReaction().true_parameters)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_data,
)

num_samples = 200

sample_indexes = jr.choice(
    key_samples, len(nuts_quasi.alpha.flatten()), shape=(num_samples,), replace=False
)

nuts_quasi_samples = jtu.tree_map(lambda x: x.flatten()[sample_indexes], nuts_quasi)

currents = vmap(fd_solver.solve)(nuts_quasi_samples)

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
        c="C1",
        alpha=0.3,
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
    alpha=0.3,
    label="95% credible interval",
)
plt.plot(fd_solver.applied_potentials, mean_current, label="Posterior mean")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.ylabel("$J$")
plt.xlabel(r"$\theta$")

plt.legend(markerscale=5)
plt.show()
