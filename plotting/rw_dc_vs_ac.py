import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use("science")

ac = np.load("./data/rw_ac.npz")
dc = np.load("./data/rw_dc.npz")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

hist_kwargs = dict(
    alpha=0.8,
    bins=50,
    density=True,
)

ax1.hist(ac["alpha"][10_000:], label="AC", **hist_kwargs)
ax1.hist(dc["alpha"][10_000:], label="DC", **hist_kwargs)

ax2.hist(ac["kappa0"][10_000:], **hist_kwargs)
ax2.hist(dc["kappa0"][10_000:], **hist_kwargs)

ax1.set_title(r"Distribution of $\alpha$")
ax2.set_title(r"Distribution of $\kappa_0$")

ax1.set_xlabel(r"$\alpha$")
ax2.set_xlabel(r"$\kappa_0$")

ax1.set_ylabel("Probability density")
ax2.set_ylabel("Probability density")


handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Posterior distributions obtained using MCLMC sampling",
    fontsize=18,
    y=0.98,
)

plt.show()
