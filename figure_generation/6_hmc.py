import gzip
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from src.diagnostics import ess_over_time
from src.params import ElectronReactionParams, param_property_names
from src.reaction import ElectronReaction

sns.set_theme()
sns.set_context("paper", font_scale=2.0)
save = False

dir = "./data/sampling/"
file = "reaction=ElectronReaction,noise=0.02,seed=100.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

nuts: ElectronReactionParams = data["nuts"]
rwmh: ElectronReactionParams = data["rwmh"]

# %% Load sampling data

true_params = ElectronReaction().true_parameters

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True, "histtype": "step", "linewidth": 1}

ax1.set_title(r"$\alpha$")
ax1.hist(nuts.alpha.flatten(), label="NUTS", **options)
ax1.hist(rwmh.alpha.flatten(), label="RWMH", **options)
ax1.axvline(x=true_params.alpha, linestyle="--", color="black", label="True Value")
ax1.set_ylabel("Density")

ax2.set_title(r"$K_0$")
ax2.hist(nuts.K0.flatten(), **options)
ax2.hist(rwmh.K0.flatten(), **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="black")

ax3.set_title(r"$\theta_f$")
ax3.hist(nuts.thetaf.flatten(), **options)
ax3.hist(rwmh.thetaf.flatten(), **options)
ax3.axvline(x=true_params.thetaf, linestyle="--", color="black")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.1, 1, 1))
if save:
    plt.savefig("./manuscript/figures/6-hmc-hist.png", dpi=1000)
plt.show()

# %%

fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
num_points = 20
x = jnp.linspace(1 / num_points, 1, num_points)
param_names = param_property_names(nuts)
headers = [r"$\alpha$", r"$K_0$", r"$\theta_f$"]

for name, header, ax in zip(param_names, headers, axs):
    ax.set_title(header)
    nuts_ess = ess_over_time(getattr(nuts, name), num_points=num_points)
    ax.plot(x, nuts_ess, label="NUTS")
    rwmh_ess = ess_over_time(getattr(rwmh, name), num_points=num_points)
    ax.plot(x, rwmh_ess, label="RWMH")

axs[0].set_ylabel("ESS")
axs[1].set_xlabel("Sample Proportion")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
if save:
    plt.savefig("./manuscript/figures/6-ess.png", dpi=1000)
plt.show()
