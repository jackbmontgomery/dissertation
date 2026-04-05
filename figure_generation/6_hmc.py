import gzip
import pickle

import blackjax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import jit

from src.params import ElectronReactionParams
from src.reaction import ElectronReaction

sns.set_theme()
sns.set_context("paper", font_scale=2.0)


# %% Load sampling data

dir = "./data/sampling/"
file = "reaction=ElectronReaction,noise=0.02,seed=42.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

nuts: ElectronReactionParams = data["nuts"]
rwmh: ElectronReactionParams = data["rwmh"]
true_params = ElectronReaction().true_parameters

# %% Plot sampling histograms

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True, "histtype": "step", "linewidth": 2}

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
# plt.savefig("./manuscript/figures/6-hmc-hist.png", dpi=1000)
plt.show()

# %% Compute the ESS

num_points = 20

nuts_chain_len = nuts.alpha.shape[1]
rwmh_chain_len = rwmh.alpha.shape[1]

nuts_end_idx = jnp.linspace(
    nuts_chain_len / num_points, nuts_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

nuts_ess = np.zeros(shape=(3, num_points))
rwmh_ess = np.zeros(shape=(3, num_points))

ess_single = jit(blackjax.diagnostics.effective_sample_size)

for i, (h_ei, r_ei) in enumerate(zip(nuts_end_idx, rwmh_end_idx)):
    nuts_ess[0, i] = ess_single(nuts.alpha[:, :h_ei])
    nuts_ess[1, i] = ess_single(nuts.K0[:, :h_ei])
    nuts_ess[2, i] = ess_single(nuts.thetaf[:, :h_ei])

    rwmh_ess[0, i] = ess_single(rwmh.alpha[:, :r_ei])
    rwmh_ess[1, i] = ess_single(rwmh.K0[:, :r_ei])
    rwmh_ess[2, i] = ess_single(rwmh.thetaf[:, :r_ei])

# %% Plot the ESS

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

ax1.set_title(r"$\alpha$")
ax2.set_title(r"$K_0$")
ax3.set_title(r"$\theta_f$")

idx = jnp.linspace(1 / num_points, 1, num_points)

ax1.plot(idx, nuts_ess[0, :], label="NUTS")
ax2.plot(idx, nuts_ess[1, :])
ax3.plot(idx, nuts_ess[2, :])

ax1.plot(idx, rwmh_ess[0, :], label="RWMH")
ax2.plot(idx, rwmh_ess[1, :])
ax3.plot(idx, rwmh_ess[2, :])

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
# plt.savefig("./manuscript/figures/6-ess.png", dpi=1000)
plt.show()

# %%

params = ["alpha", "K0", "thetaf"]

latex_params = [r"$\alpha$", r"$K_0$", r"$\theta_f$"]

for param, latex_param in zip(params, latex_params):
    hmc_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(nuts, param))
    )
    rwmh_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(rwmh, param))
    )
    print(f"{latex_param} & {hmc_rhat:.4f} & {rwmh_rhat:.4f} \\\\")
print(r"\end{tabular}")
