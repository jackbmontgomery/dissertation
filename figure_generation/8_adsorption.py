import gzip
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.fdm import AdsorptionReactionNewtonFDSolver
from src.params import AdsorptionReactionParams
from src.plotting import plot_ess_over_time, plot_potential_scale_reduction_over_time
from src.reaction import AdsorptionReaction
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2)

# %%

voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=50)

fd_solver = AdsorptionReactionNewtonFDSolver(voltammetry)
params = AdsorptionReaction().true_parameters

solution, current = fd_solver.solve(params)

plt.plot(fd_solver.applied_potentials, current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

# %%

electron_optim = np.load(
    "./data/optimisation/reaction=AdsorptionReaction,noise=0.02,seed=0.npz"
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

plt.ylim(mode_ld * 20, -2 * mode_ld)

plt.axhline(y=mode_ld * 2, linestyle="--", c="C3")
plt.legend()

plt.ylabel("Log Density")
plt.xlabel("Iteration")
plt.tight_layout()
plt.savefig("./manuscript/figures/8-optim.png", dpi=1000)
plt.show()

# %% Sampling

dir = "./data/sampling"
file = "reaction=AdsorptionReaction,noise=0.02,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

hmc: AdsorptionReactionParams = data["nuts"]
rwmh: AdsorptionReactionParams = data["rwmh"]

true_params: AdsorptionReactionParams = AdsorptionReaction().true_parameters

fig, axs = plt.subplots(3, 3, figsize=(10, 6))
options = {"density": True, "bins": 50, "alpha": 0.8, "histtype": "step"}

axs[0, 0].hist(hmc.alpha_sol.flatten(), label="HMC", **options)
axs[0, 0].hist(rwmh.alpha_sol.flatten(), label="RWMH", **options)
axs[0, 0].axvline(
    x=true_params.alpha_sol, linestyle="--", color="black", label="True Value"
)


axs[0, 1].hist(hmc.K0_sol.flatten(), **options)
axs[0, 1].hist(rwmh.K0_sol.flatten(), **options)
axs[0, 1].axvline(
    x=true_params.K0_sol, linestyle="--", color="black", label="True Value"
)

axs[0, 2].hist(hmc.thetaf_sol.flatten(), **options)
axs[0, 2].hist(rwmh.thetaf_sol.flatten(), **options)
axs[0, 2].axvline(
    x=true_params.thetaf_sol, linestyle="--", color="black", label="True Value"
)

axs[1, 0].hist(hmc.alpha_ads.flatten(), **options)
axs[1, 0].hist(rwmh.alpha_ads.flatten(), **options)
axs[1, 0].axvline(
    x=true_params.alpha_ads, linestyle="--", color="black", label="True Value"
)

axs[1, 1].hist(hmc.K0_ads.flatten(), **options)
axs[1, 1].hist(rwmh.K0_ads.flatten(), **options)
axs[1, 1].axvline(
    x=true_params.K0_ads, linestyle="--", color="black", label="True Value"
)

axs[1, 2].hist(hmc.K_A_ads.flatten(), **options)
axs[1, 2].hist(rwmh.K_A_ads.flatten(), **options)
axs[1, 2].axvline(
    x=true_params.K_A_ads, linestyle="--", color="black", label="True Value"
)

axs[2, 0].hist(hmc.K_A_des.flatten(), **options)
axs[2, 0].hist(rwmh.K_A_des.flatten(), **options)
axs[2, 0].axvline(
    x=true_params.K_A_des, linestyle="--", color="black", label="True Value"
)

axs[2, 1].hist(hmc.K_B_ads.flatten(), **options)
axs[2, 1].hist(rwmh.K_B_ads.flatten(), **options)
axs[2, 1].axvline(
    x=true_params.K_B_ads, linestyle="--", color="black", label="True Value"
)

axs[2, 2].hist(hmc.K_B_des.flatten(), **options)
axs[2, 2].hist(rwmh.K_B_des.flatten(), **options)
axs[2, 2].axvline(
    x=true_params.K_B_des, linestyle="--", color="black", label="True Value"
)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()


# %%

dir = "./data/sampling"
file = "reaction=AdsorptionReaction,noise=0.02,seed=0.pkl.gz"

with gzip.open(f"{dir}/{file}", "rb") as f:
    data = pickle.load(f)

fig = plot_ess_over_time(data, 20, ignore_prop=["thetaf_ads"])
plt.show()
