import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

# %% Noise 0.25

electron_optim = np.load(
    "./data/optimisation/reaction=ElectronReaction,noise=0.25,seed=0.npz"
)
adam_ld = electron_optim["adam_ld"]
cmaes_ld = electron_optim["cmaes_ld"]
mode_ld = electron_optim["mode_logdensity"]

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

plt.axhline(y=mode_ld * 2, label="Delta100", linestyle="--", c="C3")

plt.ylabel("Log Density")
plt.xlabel("Iteration")
plt.tight_layout()
plt.ylim(mode_ld * 10, 0)
plt.show()

# %% Noise 0.5

electron_optim = np.load(
    "./data/optimisation/reaction=ElectronReaction,noise=0.5,seed=0.npz"
)
adam_ld = electron_optim["adam_ld"]
cmaes_ld = electron_optim["cmaes_ld"]
mode_ld = electron_optim["mode_logdensity"]

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

plt.axhline(y=mode_ld * 2, label="Delta100", linestyle="--", c="C3")

plt.ylabel("Log Density")
plt.xlabel("Iteration")
plt.tight_layout()
plt.ylim(mode_ld * 10, 0)
plt.show()
