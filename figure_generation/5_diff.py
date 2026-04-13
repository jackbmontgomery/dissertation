import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

# %% Algorithm noise comparison

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
noise_levels = [0.01, 0.02]

for n, ax in zip(noise_levels, axs):
    ax.set_title(rf"$\eta={n}$")
    electron_optim = np.load(
        f"./data/optimisation/reaction=ElectronReaction,noise={n},seed=0.npz"
    )

    adam_ld = electron_optim["adam_ld"]
    cmaes_ld = electron_optim["cmaes_ld"]
    mode_ld = electron_optim["mode_logdensity"]

    iterations_adam = jnp.arange(1, adam_ld.shape[1] + 1) / adam_ld.shape[1]
    iterations_cmaes = jnp.arange(1, cmaes_ld.shape[1] + 1) / cmaes_ld.shape[1]

    a_ld_mean = jnp.mean(adam_ld, axis=0)
    a_ld_std = jnp.std(adam_ld, axis=0)

    c_ld_mean = jnp.mean(cmaes_ld, axis=0)
    c_ld_std = jnp.std(cmaes_ld, axis=0)

    ax.plot(iterations_adam, a_ld_mean, label="ADAM")
    ax.fill_between(
        iterations_adam, a_ld_mean - a_ld_std, a_ld_mean + a_ld_std, alpha=0.5
    )

    ax.plot(iterations_cmaes, c_ld_mean, label="CMA-ES")
    ax.fill_between(
        iterations_cmaes, c_ld_mean - c_ld_std, c_ld_mean + c_ld_std, alpha=0.5
    )

    ax.set_ylim(mode_ld * 5, -mode_ld)

    ax.axhline(y=mode_ld * 2.5, linestyle="--", c="black")


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

axs[0].set_ylabel("Log Density")
axs[0].set_xlabel("Proportion of run")
axs[1].set_xlabel("Proportion of run")
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/5-optim.png", dpi=1000)
plt.show()
