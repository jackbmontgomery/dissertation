import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import PRNGKeyArray

from src.experiment import CyclicMacroBand1D
from src.fdm import ButlerVolmerFDMDiscretisation1D, fdm_implicit_solve
from src.pde_parameters import ButlerVolmerParameters

key = jax.random.key(0)

theta_i = 10.0
theta_v = -10.0
sigma = 100.0

dtheta = 0.1
dx = 1e-2
dt = dtheta / sigma

experiment = CyclicMacroBand1D(theta_i, theta_v, sigma)

X = jnp.linspace(
    experiment.x_min, experiment.x_max, int((experiment.x_max - experiment.x_min) / dx)
)

T = jnp.linspace(
    experiment.t_min, experiment.t_max, int((experiment.t_max - experiment.t_min) / dt)
)

print(X.shape, T.shape)

potentials = jax.vmap(experiment.potential)(T)

pde_discretisation = ButlerVolmerFDMDiscretisation1D(X)
c_init = jnp.ones_like(X)


def simulate(
    params: ButlerVolmerParameters,
    c_init=c_init,
    pde_discretisation=pde_discretisation,
    dx=dx,
    potentials=potentials,
):
    _, current = fdm_implicit_solve(params, c_init, pde_discretisation, dx, potentials)
    return current


def generate_samples(params: ButlerVolmerParameters, key: PRNGKeyArray):
    current = simulate(target_params)
    noisy_current = jr.normal(key, shape=current.shape) * jnp.sqrt(target_params[2])
    return noisy_current


target_params: ButlerVolmerParameters = jnp.array([0.8, 2, 0.1])
key_samples = jr.split(key, 5)
current_samples = jax.vmap(generate_samples, in_axes=(None, 0))(
    target_params, key_samples
)
current_samples = jr.normal(key, shape=current_samples.shape) * jnp.sqrt(
    target_params[2]
)


def log_density(theta, samples=current_samples):
    pred = simulate(theta)
    return -len(samples) * jnp.log(theta[2]) - (jnp.sum(samples - pred) ** 2) / (
        2 * theta[2]
    )


inv_mass_matrix = jnp.array([0.1, 0.1, 0.1])
num_integration_steps = 50
step_size = 1e-5

hmc = blackjax.hmc(log_density, step_size, inv_mass_matrix, num_integration_steps)
init_params: ButlerVolmerParameters = jnp.array([0.7, 1.5, 0.2])
initial_state = hmc.init(init_params)
hmc_kernel = jax.jit(hmc.step)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        # jax.debug.print("{i}", i=info)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


rng_key, sample_key = jax.random.split(key)
states = inference_loop(sample_key, hmc_kernel, initial_state, 200)

mcmc_samples = states.position
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))
ax1.hist(mcmc_samples[:, 0], bins=25)
ax1.set_xlabel("alpha")

ax2.hist(mcmc_samples[:, 1], bins=25)
ax2.set_xlabel("k0")

ax3.hist(mcmc_samples[:, 2], bins=25)
ax3.set_xlabel("sigma")
plt.tight_layout()
plt.show()
