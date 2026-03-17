import gzip
import pickle

import blackjax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import jit

from src.params import ElectronReactionParams
from src.utils import batch_ess

sns.set_theme()
sns.set_context("paper", font_scale=1.5)


# %%
dir = "./data/sampling"
file = "reaction=ElectronReaction,noise=0.25,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

hmc: ElectronReactionParams = data["hmc"]
rwmh: ElectronReactionParams = data["rwmh"]

# %%
num_points = 20

hmc_chain_len = hmc.alpha.shape[1]
rwmh_chain_len = rwmh.alpha.shape[1]

hmc_end_idx = jnp.linspace(
    hmc_chain_len / num_points, hmc_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

hmc_ess = np.zeros(shape=(3, num_points))
rwmh_ess = np.zeros(shape=(3, num_points))

ess_single = jit(blackjax.diagnostics.effective_sample_size)

for i, (h_ei, r_ei) in enumerate(zip(hmc_end_idx, rwmh_end_idx)):
    hmc_ess[0, i] = ess_single(hmc.alpha[:, :h_ei])
    hmc_ess[1, i] = ess_single(hmc.K0[:, :h_ei])
    hmc_ess[2, i] = ess_single(hmc.Ef[:, :h_ei])

    rwmh_ess[0, i] = ess_single(rwmh.alpha[:, :r_ei])
    rwmh_ess[1, i] = ess_single(rwmh.K0[:, :r_ei])
    rwmh_ess[2, i] = ess_single(rwmh.Ef[:, :r_ei])

# %%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

ax1.set_title(r"$\alpha$")
ax2.set_title(r"$K_0$")
ax3.set_title(r"$\theta_f$")

idx = jnp.linspace(1 / num_points, 1, num_points)

ax1.plot(idx, hmc_ess[0, :], label="HMC")
ax2.plot(idx, hmc_ess[1, :])
ax3.plot(idx, hmc_ess[2, :])

ax1.plot(idx, rwmh_ess[0, :], label="RWMH")
ax2.plot(idx, rwmh_ess[1, :])
ax3.plot(idx, rwmh_ess[2, :])

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()

# %%

params = ["alpha", "K0", "Ef"]

latex_params = [r"$\alpha$", r"$K_0$", r"$E_f$"]

for param, latex_param in zip(params, latex_params):
    hmc_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(hmc, param))
    )
    rwmh_rhat = float(
        blackjax.diagnostics.potential_scale_reduction(getattr(rwmh, param))
    )
    print(f"{latex_param} & {hmc_rhat:.4f} & {rwmh_rhat:.4f} \\\\")
print(r"\end{tabular}")

# %%

num_points = 20

hmc_chain_len = hmc.alpha.shape[1]
rwmh_chain_len = rwmh.alpha.shape[1]

hmc_end_idx = jnp.linspace(
    hmc_chain_len / num_points, hmc_chain_len, num_points, dtype=jnp.int32
)

rwmh_end_idx = jnp.linspace(
    rwmh_chain_len / num_points, rwmh_chain_len, num_points, dtype=jnp.int32
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

idx = jnp.linspace(1 / num_points, 1, num_points)
ax1.plot(idx, batch_ess(hmc.alpha, hmc_end_idx), label="HMC")
ax2.plot(idx, batch_ess(hmc.K0, hmc_end_idx))
ax3.plot(idx, batch_ess(hmc.Ef, hmc_end_idx))

ax1.plot(idx, batch_ess(rwmh.alpha, rwmh_end_idx), label="RWMH")
ax2.plot(idx, batch_ess(rwmh.alpha, rwmh_end_idx))
ax3.plot(idx, batch_ess(rwmh.alpha, rwmh_end_idx))

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.show()
