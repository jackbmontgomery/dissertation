import gzip
import pickle

import blackjax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from src.params import HeterogenousReactionParams
from src.reaction import HeterogeneousReaction

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

# %% Optimisation

hetero_optim = np.load(
    "./data/optimisation/reaction=HeterogeneousReaction,noise=0.25,seed=0.npz"
)

adam_ld = hetero_optim["adam_ld"]
cmaes_ld = hetero_optim["cmaes_ld"]
mode_ld = hetero_optim["mode_logdensity"]

for a_ld, c_ld in zip(adam_ld, cmaes_ld):
    plt.plot(a_ld, c="C0", label="ADAM")
    plt.plot(c_ld, c="C1", label="CMA-ES")

legend_elements = [
    Patch(facecolor="C0", label="ADAM"),
    Patch(facecolor="C1", label="CMA-ES"),
]

plt.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=2,
    frameon=False,
)

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.ylim(mode_ld * 10, 0)
plt.show()

# %% Sampling

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.25,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

hmc: HeterogenousReactionParams = data["hmc"]
rwmh: HeterogenousReactionParams = data["rwmh"]

true_params: HeterogenousReactionParams = HeterogeneousReaction().true_parameters

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_K1 = fig.add_subplot(gs[0, 1])
ax_Ef1 = fig.add_subplot(gs[0, 2])
ax_a2 = fig.add_subplot(gs[1, 0])
ax_K2 = fig.add_subplot(gs[1, 1])
ax_Ef2 = fig.add_subplot(gs[1, 2])

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])

options = {"density": True, "bins": 50, "alpha": 0.8, "histtype": "step"}

ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.hist(hmc.alpha_1.flatten(), label="HMC", **options)
ax_a1.hist(rwmh.alpha_1.flatten(), **options, label="RWMH")
ax_a1.axvline(x=true_params.alpha_1, linestyle="--", color="black", label="True Value")

ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.hist(hmc.alpha_2.flatten(), **options)
ax_a2.hist(rwmh.alpha_2.flatten(), **options)
ax_a2.axvline(x=true_params.alpha_2, linestyle="--", color="black")

ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.hist(hmc.K0_1.flatten(), **options)
ax_K1.hist(rwmh.K0_1.flatten(), **options)
ax_K1.axvline(x=true_params.K0_1, linestyle="--", color="black")

ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.hist(hmc.K0_2.flatten(), **options)
ax_K2.hist(rwmh.K0_2.flatten(), **options)
ax_K2.axvline(x=true_params.K0_2, linestyle="--", color="black")

ax_Ef1.set_title(r"$E_f^{(1)}$")
ax_Ef1.hist(hmc.Ef_1.flatten(), **options)
ax_Ef1.hist(rwmh.Ef_1.flatten(), **options)
ax_Ef1.axvline(x=true_params.Ef_1, linestyle="--", color="black")

ax_Ef2.set_title(r"$E_f^{(2)}$")
ax_Ef2.hist(hmc.Ef_2.flatten(), **options)
ax_Ef2.hist(rwmh.Ef_2.flatten(), **options)
ax_Ef2.axvline(x=true_params.Ef_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(hmc.K_het.flatten(), **options)
ax_Khet.hist(rwmh.K_het.flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)

plt.show()

# %% ESS

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.25,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

hmc: HeterogenousReactionParams = data["hmc"]
rwmh: HeterogenousReactionParams = data["rwmh"]


num_points = 20

hmc_chain_len = hmc.alpha_1.shape[1]
rwmh_chain_len = rwmh.alpha_1.shape[1]

hmc_end_idx = jnp.linspace(
    hmc_chain_len / num_points, hmc_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

hmc_ess = np.zeros(shape=(7, num_points))
rwmh_ess = np.zeros(shape=(7, num_points))

for i, (h_ei, r_ei) in enumerate(zip(hmc_end_idx, rwmh_end_idx)):
    hmc_ess[0, i] = blackjax.diagnostics.effective_sample_size(hmc.alpha_1[:, :h_ei])
    hmc_ess[1, i] = blackjax.diagnostics.effective_sample_size(hmc.K0_1[:, :h_ei])
    hmc_ess[2, i] = blackjax.diagnostics.effective_sample_size(hmc.Ef_1[:, :h_ei])
    hmc_ess[3, i] = blackjax.diagnostics.effective_sample_size(hmc.alpha_2[:, :h_ei])
    hmc_ess[4, i] = blackjax.diagnostics.effective_sample_size(hmc.K0_2[:, :h_ei])
    hmc_ess[5, i] = blackjax.diagnostics.effective_sample_size(hmc.Ef_2[:, :h_ei])
    hmc_ess[6, i] = blackjax.diagnostics.effective_sample_size(hmc.K_het[:, :h_ei])

    rwmh_ess[0, i] = blackjax.diagnostics.effective_sample_size(rwmh.alpha_1[:, :r_ei])
    rwmh_ess[1, i] = blackjax.diagnostics.effective_sample_size(rwmh.K0_1[:, :r_ei])
    rwmh_ess[2, i] = blackjax.diagnostics.effective_sample_size(rwmh.Ef_1[:, :r_ei])
    rwmh_ess[3, i] = blackjax.diagnostics.effective_sample_size(rwmh.alpha_2[:, :r_ei])
    rwmh_ess[4, i] = blackjax.diagnostics.effective_sample_size(rwmh.K0_2[:, :r_ei])
    rwmh_ess[5, i] = blackjax.diagnostics.effective_sample_size(rwmh.Ef_2[:, :r_ei])
    rwmh_ess[6, i] = blackjax.diagnostics.effective_sample_size(rwmh.K_het[:, :r_ei])

# %%  Plot ESS

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.set_ylabel("ESS")

ax_K1 = fig.add_subplot(gs[0, 1])
ax_K1.set_title(r"$K_0^{(1)}$")

ax_Ef1 = fig.add_subplot(gs[0, 2])
ax_Ef1.set_title(r"$E_f^{(1)}$")

ax_a2 = fig.add_subplot(gs[1, 0])
ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.set_xlabel("Sample Proportion")
ax_a2.set_ylabel("ESS")

ax_K2 = fig.add_subplot(gs[1, 1])
ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.set_xlabel("Sample Proportion")

ax_Ef2 = fig.add_subplot(gs[1, 2])
ax_Ef2.set_title(r"$E_f^{(2)}$")
ax_Ef2.set_xlabel("Sample Proportion")

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])
ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.set_xlabel("Sample Proportion")

axs = [ax_a1, ax_K1, ax_Ef1, ax_a2, ax_K2, ax_Ef2, ax_Khet]

idx = jnp.linspace(1 / num_points, 1, num_points)

for i, ax in enumerate(axs):
    ax.plot(idx, hmc_ess[i, :], label="HMC")
    ax.plot(idx, rwmh_ess[i, :], label="RWMH")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)
plt.show()
