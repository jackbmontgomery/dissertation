import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns

from src.fdm import ElectronReactionFDSolver
from src.optimisers import make_adam_optimise, make_cmaes_optimise
from src.params import ElectronReactionParams
from src.reaction import ElectronReaction
from src.sampling import RWMHSampler
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2)

key = jr.key(0)
save = False

# %% RWMH vs ADAM

key_samples, key_init, key_sampling, key_cmaes = jr.split(key, 4)

voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=100)
fd_solver = ElectronReactionFDSolver(voltammetry)
reaction = ElectronReaction()

true_params = reaction.true_parameters
base_current = fd_solver.solve(true_params)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_samples,
)


def logdensity_fn(params: ElectronReactionParams):
    current = fd_solver.solve(params)
    return -jnp.sum((experimental_samples - current) ** 2)


init_params = reaction.create_init_params(key_init, 1)

rwmh = RWMHSampler(logdensity_fn, 750, 1)

rwmh_params = {"random_step": blackjax.mcmc.random_walk.normal(jnp.repeat(0.01, 3))}

samples, infos = rwmh.run(init_params, rwmh_params, key=key_sampling)
samples: ElectronReactionParams = samples

adam_params = {"learning_rate": 1e-1}
adam_optimise = make_adam_optimise(50, logdensity_fn, **adam_params)

_, adam_ld, adam_pp = adam_optimise(init_params)
adam_pp: ElectronReactionParams = adam_pp

cmaes_params = {"population_size": 4}
cmaes_optimise = make_cmaes_optimise(50, logdensity_fn, **cmaes_params)

_, cmaes_ld, cmaes_pp = cmaes_optimise(init_params, key_cmaes)
cmaes_pp: ElectronReactionParams = cmaes_pp

# %%

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(10, 3), sharex=True, sharey=True, layout="constrained"
)

idx_adam = jnp.arange(len(adam_pp.alpha.flatten()))
idx_cmaes = jnp.arange(len(cmaes_pp.alpha.flatten()))
idx_hmc = jnp.arange(len(samples.alpha.flatten()))

# Normalise all indices to [0, 1] for a shared colormap
norm = plt.Normalize(vmin=0, vmax=1)

sc1 = ax1.scatter(
    adam_pp.alpha.flatten(),
    adam_pp.K0.flatten(),
    c=idx_adam / idx_adam.max(),
    cmap="viridis",
    norm=norm,
    s=1,
)
sc2 = ax2.scatter(
    cmaes_pp.alpha.flatten(),
    cmaes_pp.K0.flatten(),
    c=idx_cmaes / idx_cmaes.max(),
    cmap="viridis",
    norm=norm,
    s=1,
)
sc3 = ax3.scatter(
    samples.alpha.flatten(),
    samples.K0.flatten(),
    c=idx_hmc / idx_hmc.max(),
    cmap="viridis",
    norm=norm,
    s=1,
)

for ax in (ax1, ax2, ax3):
    ax.scatter(
        [true_params.alpha],
        [true_params.K0],
        marker="x",
        s=75.0,
        c="black",
        label="True Value",
        zorder=5,
    )

ax1.set_title("ADAM")
ax1.set_xlabel(r"$\alpha$")
ax2.set_title("CMA-ES")
ax2.set_xlabel(r"$\alpha$")
ax3.set_title("RWMH")
ax3.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$K_0$")
fig.colorbar(
    sc3,
    ax=[ax1, ax2, ax3],
    label="Index",
    location="right",
)
if save:
    fig.savefig("./manuscript/figures/5-burn-in-scatter.png", dpi=1000)
plt.show()
