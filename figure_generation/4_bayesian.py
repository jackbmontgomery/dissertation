import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.reaction import ElectronReaction
from src.sampling import RWMHSampler
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=2.0)

key = jr.key(0)

# %% Noisy Samples
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

samples_key, sampling_key, key = jr.split(key, 3)
voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)
true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    thetaf=jnp.array(0.0),
)
init_params = ElectronReactionParams(
    alpha=jnp.array(0.5), K0=jnp.array(15.0), thetaf=jnp.array(0.5)
)

base_current = fd_solver.solve(true_params)

all_noise = [0.01, 0.02]

for noise, ax in zip(all_noise, axs):
    ax.set_title(rf"$\eta = {noise}$")
    experimental_samples = generate_noisy_samples(
        5, base_current, noise, key=samples_key
    )

    for i, sample in enumerate(experimental_samples):
        ax.plot(fd_solver.applied_potentials, sample, alpha=0.6)

axs[0].set_ylabel(r"$J$")
axs[0].set_xlabel(r"$\theta$")
axs[1].set_xlabel(r"$\theta$")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./manuscript/figures/4-noisy-data.png", dpi=1000)
plt.show()

# %% Random-Walk Metropolis Hasting with no burn-in

key_samples, key_init, key_sampling = jr.split(key, 3)
voltammetry = CyclicDC()
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

rwmh = RWMHSampler(logdensity_fn, 1000, 1)

rwmh_params = {"random_step": blackjax.mcmc.random_walk.normal(jnp.repeat(0.01, 3))}

samples, _ = rwmh.run(init_params, rwmh_params, key=key_sampling)
samples: ElectronReactionParams = samples

# %% Hist plot with no burn in

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 30, "density": True}

ax1.set_title(r"$\alpha$")
ax1.hist(samples.alpha.flatten(), **options, label="Samples")
ax1.axvline(
    x=true_params.alpha, linestyle="--", color="C3", linewidth=2.0, label="True Value"
)
ax1.set_ylabel("Density")
ax2.set_title(r"$K_0$")
ax2.hist(samples.K0.flatten(), **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="C3", linewidth=2.0)
ax3.set_title(r"$\theta_f$")
ax3.hist(samples.thetaf.flatten(), **options)
ax3.axvline(x=true_params.thetaf, linestyle="--", color="C3", linewidth=2.0)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./manuscript/figures/4-rwmh-hist.png", dpi=1000)
plt.show()

# %% Scatter plot for no burn-in

idx = jnp.arange(len(samples.alpha.flatten()))
plt.scatter(samples.alpha.flatten(), samples.K0.flatten(), c=idx, cmap="viridis", s=0.5)
plt.colorbar(label="Sample Index")
plt.scatter(
    [true_params.alpha],
    [true_params.K0],
    marker="x",
    s=75.0,
    label="True Value",
    c="C3",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$K_0$")
plt.tight_layout()
plt.savefig("./manuscript/figures/4-rwmh-scatter.png", dpi=1000)
plt.show()
