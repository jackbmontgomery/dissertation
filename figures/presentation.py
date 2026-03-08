# %% Imports
import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from jax import jit, vmap
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))

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
plt.show()
plt.close()

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

ax2.set_title(r"$E_f$")
ax2.set_xlabel(r"$\theta$")
ax2.legend()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.close()


# %% Noisy Samples
true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.5),
)


plt.figure(figsize=(8, 6))

voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)
_, base_current = fd_solver.solve(true_params)

for sigma in [0.5, 0.25, 0.1]:
    noise_key, key = jr.split(key)
    noisy_current = generate_noisy_samples(1, base_current, sigma=sigma, key=noise_key)[
        0
    ]
    plt.plot(fd_solver.applied_potentials, noisy_current)


plt.ylabel(r"$J$")
plt.xlabel(r"$\theta$")

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
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

jit_step = jit(rw.step)
init_states = rw.init(init_params)
states, infos = inference_loop(sampling_key, jit_step, init_states, 5_000)
samples = states.position

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True}

ax1.set_title(r"$\alpha$")
ax1.hist(samples.alpha, **options)
ax1.axvline(x=true_params.alpha, linestyle="--", color="black")
ax2.set_title(r"$K_0$")
ax2.hist(samples.K0, **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="black")
ax3.set_title(r"$\theta_f^0$")
ax3.hist(samples.Ef, **options)
ax3.axvline(x=true_params.Ef, linestyle="--", color="black")
plt.show()

# %% Scatter plot for no burn-in
idx = jnp.arange(len(samples.alpha))
plt.scatter(samples.alpha, samples.K0, c=idx, cmap="viridis", s=0.5)
plt.colorbar(label="Sample Index")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$K_0$")
plt.tight_layout()
plt.show()


# %% Random-Walk Metropolis Hasting gradient-burn in

minimised_init_params, log_likelihood, params_path = adam_minimise(
    init_params, learning_rate=1e-1, num_steps=50, log_density=log_density
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

n_mh = 100
n_gd = len(params_path.alpha)

mh_cost_per_step = 1
gd_cost_per_step = 2

mh_cumulative_cost = jnp.arange(n_mh) * mh_cost_per_step  # 0, 1, 2, ...99
gd_cumulative_cost = jnp.arange(n_gd) * gd_cost_per_step  # 0, 2, 4, ...

vmax = max(mh_cumulative_cost[-1], gd_cumulative_cost[-1])

sc1 = ax1.scatter(
    samples.alpha[:n_mh],
    samples.K0[:n_mh],
    c=mh_cumulative_cost,
    cmap="viridis",
    s=5.0,
    vmin=0,
    vmax=vmax,
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
    c=gd_cumulative_cost,
    cmap="viridis",
    s=5.0,
    vmin=0,
    vmax=vmax,
)
ax2.set_title("Gradient Descent (Adam)")
ax2.set_xlabel(r"$\alpha$")

# Single shared colourbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])

sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax, label="Cumulative compute")

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %% MH sampling with minimised_init_params

init_states = rw.init(minimised_init_params)
states, infos = inference_loop(sampling_key, jit_step, init_states, 1_000)
minimised_samples = states.position

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
options = {"bins": 50, "density": True, "histtype": "step"}

ax1.set_title(r"$\alpha$")
ax1.hist(samples.alpha, **options, label="MH Burn-in")
ax1.hist(minimised_samples.alpha, **options, label="GD Burn-in")
ax1.axvline(x=true_params.alpha, linestyle="--", color="black")
ax2.set_title(r"$K_0$")
ax2.hist(samples.K0, **options)
ax2.hist(minimised_samples.K0, **options)
ax2.axvline(x=true_params.K0, linestyle="--", color="black")
ax3.set_title(r"$\theta_f^0$")
ax3.hist(samples.Ef, **options)
ax3.hist(minimised_samples.Ef, **options)
ax3.axvline(x=true_params.Ef, linestyle="--", color="black")

plt.show()
