import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from jax import jit

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.sampling import inference_loop
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("paper", font_scale=1.5)

key = jr.key(0)

# %% Noisy Samples
samples_key, sampling_key, key = jr.split(key, 3)
cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)
true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.0),
)
init_params = ElectronReactionParams(
    alpha=jnp.array(0.5), K0=jnp.array(15.0), Ef=jnp.array(0.5)
)

_, base_current = fd_solver.solve(true_params)
experimental_samples = generate_noisy_samples(5, base_current, 0.25, key=samples_key)

for i, sample in enumerate(experimental_samples):
    plt.plot(fd_solver.applied_potentials, sample, label=i)

plt.ylabel(r"$J$")
plt.xlabel(r"$\theta$")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./write_up/figures/4-noisy-data.png", dpi=1000)
plt.show()

# %% Random-Walk Metropolis Hasting with no burn-in

samples_key, sampling_key, key = jr.split(key, 3)
cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)
true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.0),
)
init_params = ElectronReactionParams(
    alpha=jnp.array(0.5), K0=jnp.array(15.0), Ef=jnp.array(0.5)
)

_, base_current = fd_solver.solve(true_params)
experimental_samples = generate_noisy_samples(10, base_current, 0.25, key=samples_key)


def log_density(params: ElectronReactionParams, samples=experimental_samples):
    _, current = fd_solver.solve(params)
    return -jnp.sum((samples - current) ** 2)


rw = blackjax.additive_step_random_walk(
    log_density, blackjax.mcmc.random_walk.normal(jnp.repeat(0.01, 3))
)

rw_jit_step = jit(rw.step)
init_states = rw.init(init_params)
states, infos = inference_loop(sampling_key, rw_jit_step, init_states, 5_000)
samples = states.position

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 25, "density": True}

ax1.set_title(r"$\alpha$")
ax1.hist(samples.alpha, **options)
ax1.axvline(x=true_params.alpha, linestyle="--", color="black", label="True Value")
ax1.set_ylabel("Density")
ax2.set_title(r"$K_0$")
ax2.hist(samples.K0, **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="black")
ax3.set_title(r"$\theta_f^0$")
ax3.hist(samples.Ef, **options)
ax3.axvline(x=true_params.Ef, linestyle="--", color="black")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("./write_up/figures/4-rwmh-hist.png", dpi=1000)
plt.show()

# %% Scatter plot for no burn-in
idx = jnp.arange(len(samples.alpha))
plt.scatter(samples.alpha, samples.K0, c=idx, cmap="viridis", s=0.5)
plt.colorbar(label="Sample Index")
plt.scatter(
    [true_params.alpha],
    [true_params.K0],
    marker="x",
    s=75.0,
    label="True Value",
    c="black",
)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$K_0$")
plt.tight_layout()
plt.savefig("./write_up/figures/4-rwmh-scatter.png", dpi=1000)
plt.show()
