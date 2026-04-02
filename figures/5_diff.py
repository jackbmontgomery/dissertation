import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

# %% Algorithm noise comparison

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
noise_levels = [0.25, 0.5]

for n, ax in zip(noise_levels, axs):
    ax.set_title(rf"$\varsigma={n}$")
    electron_optim = np.load(
        f"./data/optimisation/reaction=ElectronReaction,noise={n},seed=0.npz"
    )

    adam_ld = electron_optim["adam_ld"]
    cmaes_ld = electron_optim["cmaes_ld"]
    mode_ld = electron_optim["mode_logdensity"]
    iterations = jnp.arange(1, adam_ld.shape[1] + 1)

    a_ld_mean = jnp.mean(adam_ld, axis=0)
    a_ld_std = jnp.std(adam_ld, axis=0)

    c_ld_mean = jnp.mean(cmaes_ld, axis=0)
    c_ld_std = jnp.std(cmaes_ld, axis=0)

    ax.plot(iterations, a_ld_mean, label="ADAM")
    ax.fill_between(iterations, a_ld_mean - a_ld_std, a_ld_mean + a_ld_std, alpha=0.5)

    ax.plot(iterations, c_ld_mean, label="CMA-ES")
    ax.fill_between(iterations, c_ld_mean - c_ld_std, c_ld_mean + c_ld_std, alpha=0.5)

    ax.set_ylim(mode_ld * 5, 0)

    ax.axhline(y=mode_ld * 2, linestyle="--", c="C3")


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

axs[0].set_ylabel("Log Density")
axs[0].set_xlabel("Iteration")
axs[1].set_xlabel("Iteration")
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/5-optim.png", dpi=1000)
plt.show()
