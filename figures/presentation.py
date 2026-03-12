# %% Imports
import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from typing import Callable

import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from jax import jit, pmap, vmap
from jax.lax import scan

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.utils import adam_minimise, generate_noisy_samples
from src.voltammetry import CyclicDC

sns.set_theme()
sns.set_context("talk")

key = jr.key(0)

# %% Applied Potential and Voltammagram

true_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(1e6), Ef=jnp.array(0.0)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

_, current = fd_solver.solve(true_params)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

ax1.plot(fd_solver.applied_potentials)
ax1.yaxis.set_inverted(True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Applied Potential")
ax1.set_xlabel("Time")

ax2.plot(fd_solver.applied_potentials, current)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.xaxis.set_inverted(True)
ax2.yaxis.set_inverted(True)
ax2.set_ylabel("Current")
ax2.set_xlabel("Applied Potential")

plt.tight_layout()
plt.savefig("./presentation/figures/cyclic-voltammetry.png", dpi=1000)
plt.show()

# %% Parameter Effect

base_params = ElectronReactionParams(
    alpha=jnp.array(0.6), K0=jnp.array(10.0), Ef=jnp.array(0.5)
)

voltammetry = CyclicDC()

fd_solver = ElectronReactionFDSolver(voltammetry)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

alpha_range = jnp.array([0.3, 0.5, 0.7])
alpha_params = ElectronReactionParams(
    alpha=alpha_range,
    K0=jnp.full_like(alpha_range, base_params.K0),
    Ef=jnp.full_like(alpha_range, base_params.Ef),
)

_, alpha_currents = vmap(fd_solver.solve)(alpha_params)
for val, current in zip(alpha_range, alpha_currents):
    ax0.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")

K0_range = jnp.array([1.0, 10.0, 50.0])

K0_params = ElectronReactionParams(
    alpha=jnp.full_like(K0_range, base_params.alpha),
    K0=K0_range,
    Ef=jnp.full_like(K0_range, base_params.Ef),
)

_, K0_currents = vmap(fd_solver.solve)(K0_params)
for val, current in zip(K0_range, K0_currents):
    ax1.plot(fd_solver.applied_potentials, current, label=f"{val:.0f}")

Ef_range = jnp.array([-1.0, 0.0, 1.0])

Ef_params = ElectronReactionParams(
    alpha=jnp.full_like(Ef_range, base_params.alpha),
    K0=jnp.full_like(Ef_range, base_params.K0),
    Ef=Ef_range,
)

_, Ef_currents = vmap(fd_solver.solve)(Ef_params)
for val, current in zip(Ef_range, Ef_currents):
    ax2.plot(fd_solver.applied_potentials, current, label=f"{val:.1f}")


ax0.set_title(r"$\alpha$")
ax0.set_ylabel(r"$J$")
ax0.set_xlabel(r"$\theta$")
ax0.legend()

ax1.set_title(r"$K_0$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()

ax2.set_title(r"$\theta_f$")
ax2.set_xlabel(r"$\theta$")
ax2.legend()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./presentation/figures/forward-problem.png", dpi=1000)
plt.show()


# %% Random-Walk Metropolis Hasting with no burn-in


def inference_loop(key, kernel, initial_state, num_samples):
    @jit
    def scan_step(state, step_key):
        state, info = kernel(step_key, state)
        return state, (state, info)

    keys = jr.split(key, num_samples)
    _, (states, infos) = scan(scan_step, initial_state, keys)

    return states, infos


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
plt.savefig("./presentation/figures/no-burn-in.png", dpi=1000)
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
plt.savefig("./presentation/figures/normal-burn-in.png", dpi=1000)
plt.show()


# %% Random-Walk Metropolis Hasting gradient-burn in

minimised_init_params, log_likelihood, params_path = adam_minimise(
    init_params, learning_rate=1e-1, num_steps=50, log_density=log_density
)

idx = jnp.arange(len(params_path.alpha))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc1 = ax1.scatter(
    samples.alpha[:50],
    samples.K0[:50],
    c=idx,
    cmap="viridis",
    s=5.0,
)

ax1.set_title("MCMC (Metropolis–Hastings)")
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$K_0$")

minimised_init_params, log_likelihood, params_path = adam_minimise(
    init_params, learning_rate=1e-1, num_steps=50, log_density=log_density
)

sc2 = ax2.scatter(
    params_path.alpha,
    params_path.K0,
    c=idx,
    cmap="viridis",
    s=5.0,
)
ax2.set_title("Gradient Descent (Adam)")
ax2.set_xlabel(r"$\alpha$")

# Single shared colourbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])

sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=50))
fig.colorbar(sm, cax=cbar_ax, label="Step")

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("./presentation/figures/burn-in-comparison.png", dpi=1000)
plt.show()

# %% MH sampling with minimised_init_params

init_states = rw.init(minimised_init_params)
states, infos = inference_loop(sampling_key, rw_jit_step, init_states, 1_000)
rw_minimised_samples = states.position


# %% HMC

inference_loop_multiple_chains: Callable = pmap(
    inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
)


def create_init_params(key, num_chains: int):
    k1, k2, k3 = jr.split(key, 3)

    alpha_vals = jnp.linspace(0.5, 0.7, num_chains)
    K0_vals = jnp.linspace(1.0, 20.0, num_chains)
    Ef_vals = jnp.linspace(0.0, 1.0, num_chains)

    alpha = jr.permutation(k1, alpha_vals)
    K0 = jr.permutation(k2, K0_vals)
    Ef = jr.permutation(k3, Ef_vals)

    return ElectronReactionParams(
        alpha=alpha,
        K0=K0,
        Ef=Ef,
    )


key_warmup, key_params, key = jr.split(key, 3)
warmup = blackjax.chees_adaptation(log_density, 8)

optim = optax.adam(1e-2)

print("--- Running Chees Warmup ---")
init_params_chains = create_init_params(key_params, 8)

(initial_states, parameters), _ = warmup.run(
    key_warmup,
    init_params_chains,
    step_size=1e-2,
    optim=optim,
    num_steps=500,
)

hmc = blackjax.dynamic_hmc(log_density, **parameters)

hmc_jit_step = jit(hmc.step)
keys_sampling = jr.split(key, 8)

hmc_states, infos = inference_loop_multiple_chains(
    keys_sampling, hmc_jit_step, initial_states, 100
)

# %% Histogram

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 25, "density": True, "histtype": "step"}

ax1.set_title(r"$\alpha$")
ax1.hist(hmc_states.position.alpha.flatten(), **options, label="HMC")
ax1.hist(rw_minimised_samples.alpha, **options, label="RW")
ax1.axvline(x=true_params.alpha, linestyle="--", color="black")
ax2.set_title(r"$K_0$")
ax2.hist(hmc_states.position.K0.flatten(), **options)
ax2.hist(rw_minimised_samples.K0, **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="black")
ax3.set_title(r"$\theta_f^0$")
ax3.hist(hmc_states.position.Ef.flatten(), **options)
ax3.hist(rw_minimised_samples.Ef, **options)
ax3.axvline(x=true_params.Ef, linestyle="--", color="black")
plt.show()

# %% Autocorrelation

import arviz as az
import matplotlib.pyplot as plt

hmc = np.load("./data/E_HMC_0.25_1000.npz")
rw = np.load("./data/E_RW_0.25_1000.npz")

param_labels = {"alpha": r"$\alpha$", "K0": r"$K_0$", "Ef": r"$\theta_f$"}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (label, display) in zip(axes, param_labels.items()):
    hmc_param = hmc[label][:, :1000]
    rw_param = rw[label][:, : hmc_param.shape[1]]

    sample_sizes = np.linspace(100, hmc_param.shape[1], 50, dtype=int)

    hmc_ess = [
        az.ess(az.convert_to_dataset({label: hmc_param[:, :n]})).to_array().values[0]
        for n in sample_sizes
    ]

    rw_ess = [
        az.ess(az.convert_to_dataset({label: rw_param[:, :n]})).to_array().values[0]
        for n in sample_sizes
    ]

    ax.plot(sample_sizes, hmc_ess, label="HMC")
    ax.plot(sample_sizes, rw_ess, label="RW-MH")
    ax.set_title(display)
    ax.set_xlabel("Samples")
    ax.set_ylabel("ESS")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

plt.tight_layout(rect=[0, 0.075, 1, 1])
plt.show()
