import gzip
import pickle
from typing import Callable

import blackjax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import jit

from src.fdm import ElectronReactionFDSolver, HeterogeneousReactionFDSolver
from src.params import ElectronReactionParams, HeterogenousReactionParams
from src.reaction import HeterogeneousReaction
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2)

# %% Current Comparison

voltammetry = CyclicDC()
elec_fd = ElectronReactionFDSolver(voltammetry)
elec_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(10.0), thetaf=jnp.array(0.0)
)
heter_fd = HeterogeneousReactionFDSolver(voltammetry)
heter_params = HeterogenousReactionParams(
    alpha_1=jnp.array(0.6),
    K0_1=jnp.array(10.0),
    thetaf_1=jnp.array(0.0),
    alpha_2=jnp.array(0.6),
    K0_2=jnp.array(10.0),
    thetaf_2=jnp.array(0.0),
    K_het=jnp.array(5.0),
)

_, elec_current = elec_fd.solve(elec_params)
_, hetero_current = heter_fd.solve(heter_params)

plt.plot(elec_fd.applied_potentials, elec_current, label="Electron")
plt.plot(heter_fd.applied_potentials, hetero_current, label="Heterogeneous")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()
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

# %% Sampling

dir = "./data/sampling"
file = "reaction=HeterogeneousReaction,noise=0.02,seed=1.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

hmc: HeterogenousReactionParams = data["hmc"]
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

ax_thetaf1.set_title(r"$E_f^{(1)}$")
ax_thetaf1.hist(hmc.thetaf_1.flatten(), **options)
ax_thetaf1.hist(rwmh.thetaf_1.flatten(), **options)
ax_thetaf1.axvline(x=true_params.thetaf_1, linestyle="--", color="black")

ax_thetaf2.set_title(r"$E_f^{(2)}$")
ax_thetaf2.hist(hmc.thetaf_2.flatten(), **options)
ax_thetaf2.hist(rwmh.thetaf_2.flatten(), **options)
ax_thetaf2.axvline(x=true_params.thetaf_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(hmc.K_het.flatten(), **options)
ax_Khet.hist(rwmh.K_het.flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)

plt.show()

# %% ESS

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

ess_single: Callable = jit(blackjax.diagnostics.effective_sample_size)

for i, (h_ei, r_ei) in enumerate(zip(hmc_end_idx, rwmh_end_idx)):
    hmc_ess[0, i] = ess_single(hmc.alpha_1[:, :h_ei])
    hmc_ess[1, i] = ess_single(hmc.K0_1[:, :h_ei])
    hmc_ess[2, i] = ess_single(hmc.thetaf_1[:, :h_ei])
    hmc_ess[3, i] = ess_single(hmc.alpha_2[:, :h_ei])
    hmc_ess[4, i] = ess_single(hmc.K0_2[:, :h_ei])
    hmc_ess[5, i] = ess_single(hmc.thetaf_2[:, :h_ei])
    hmc_ess[6, i] = ess_single(hmc.K_het[:, :h_ei])

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
    ax.plot(idx, hmc_ess[i, :], label="HMC")
    ax.plot(idx, rwmh_ess[i, :], label="RWMH")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=2)
plt.show()


# %%

params = ["alpha_1", "K0_1", "thetaf_1", "alpha_2", "K0_2", "thetaf_2", "K_het"]

for param in params:
    hmc_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(hmc, param))
    )
    rwmh_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(rwmh, param))
    )
    print(f"{hmc_rhat:.4f} & {rwmh_rhat:.4f}")
