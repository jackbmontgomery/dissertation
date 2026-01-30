import blackjax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.lax import scan

from src.experiment import CyclicMacroBand1D
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    discretise_experiment,
)
from src.pde_parameters import ButlerVolmerInverseParameters, bv_inverse_to_physical
from src.sampling import generate_noisy_samples
from src.simulate import create_fdm_current_simulator

key = jr.key(0)

experiment = CyclicMacroBand1D()
dx = 1e-3
T, X = discretise_experiment(experiment, dx=dx)

print(f"T:{X.shape},X:{T.shape}")

potentials = vmap(experiment.potential)(T)

fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X)
c_init = jnp.ones_like(X)

simulate_current = create_fdm_current_simulator(
    c_init, potentials, fdm_discretisation, dx
)

params = ButlerVolmerInverseParameters(a=jnn.sigmoid(0.6), k0=jnp.log(100.0))

phy_params = bv_inverse_to_physical(params)

n_samples = 10
sigma = 0.01

samples = generate_noisy_samples(
    n_samples, simulate_current, phy_params, sigma, key=key
)


def log_density(params: ButlerVolmerInverseParameters, samples=samples):
    phy_params = bv_inverse_to_physical(params)
    pred = simulate_current(phy_params)
    return -jnp.sum((samples - pred) ** 2)


barker = blackjax.mcmc.barker.as_top_level_api(log_density, 0.1)

initial_state = barker.init(params)

kernel = jit(barker.step)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, state

    keys = jr.split(rng_key, num_samples)
    _, states = scan(one_step, initial_state, keys)

    return states


rng_key, sample_key = jr.split(key)
states = inference_loop(sample_key, kernel, initial_state, 100)

states.position.a.block_until_ready()
mcmc_samples = vmap(bv_inverse_to_physical)(states.position)

n_bins = 50

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

ax1.hist(mcmc_samples.alpha, bins=n_bins)
ax1.set_xlabel("alpha")

ax2.hist(mcmc_samples.kappa0, bins=n_bins)
ax2.set_xlabel("k0")

plt.tight_layout()
plt.show()
