import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from time import perf_counter
from typing import Callable

import blackjax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit, pmap, vmap
from jaxtyping import Array, PRNGKeyArray

from src.experiment import CyclicMacroBand1D, LinearSweepDCMacroBand
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    uniform_discretise,
)
from src.pde_parameters import ButlerVolmerInverseParameters, bv_inverse_to_physical
from src.sampling import _inference_loop, generate_noisy_samples
from src.simulate import create_fdm_current_simulator


def parallel_rw_sampling(
    sample_keys: PRNGKeyArray,
    n_samples_per_chain: int,
    initial_parameters: ButlerVolmerInverseParameters,
    log_density: Callable[[ButlerVolmerInverseParameters, Array], Array],
    sigma: Array = jnp.array([0.1, 0.1]),
):
    rw = blackjax.additive_step_random_walk(
        log_density, blackjax.mcmc.random_walk.normal(sigma)
    )

    inference_loop_multiple_chains = pmap(
        _inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
    )

    initial_states = vmap(rw.init, in_axes=(0))(initial_parameters)

    jit_step = jit(rw.step)

    pmap_states = inference_loop_multiple_chains(
        sample_keys, jit_step, initial_states, n_samples_per_chain
    )

    return pmap_states


key = jr.key(0)

experiment = LinearSweepDCMacroBand()
dx = 1e-2
T, X = uniform_discretise(experiment, dx=dx)
dt = T[1] - T[0]

print(f"T:{X.shape},X:{T.shape}")

potentials = vmap(experiment.potential)(T)

fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, dt)
c_init = jnp.ones_like(X)

simulate_current = create_fdm_current_simulator(
    c_init, potentials, fdm_discretisation, dx
)

sample_params = ButlerVolmerInverseParameters(a=jnn.sigmoid(0.6), k0=jnp.log(100.0))

phy_params = bv_inverse_to_physical(sample_params)

samples = generate_noisy_samples(100, simulate_current, phy_params, 0.01, key=key)


def log_density(params: ButlerVolmerInverseParameters, samples=samples):
    phy_params = bv_inverse_to_physical(params)
    pred = simulate_current(phy_params)
    return -jnp.sum((samples - pred) ** 2)


num_chains = multiprocessing.cpu_count()

total_samples = 10_000
n_samples_per_chain = total_samples // num_chains

keys = jr.split(key, num_chains)

chain_params = ButlerVolmerInverseParameters(
    a=jnp.full((num_chains,), jnn.sigmoid(0.6)),
    k0=jnp.full((num_chains,), jnp.log(100.0)),
)

start_time = perf_counter()
print("Sampling Starting:", start_time)

states = parallel_rw_sampling(keys, n_samples_per_chain, chain_params, log_density)

states.position.a.block_until_ready()
end_time = perf_counter()
print("Sampling Done:", end_time)
print("Time Taken:", end_time - start_time)

alpha = jnn.sigmoid(states.position.a.flatten())
kappa0 = jnp.exp(states.position.k0.flatten())
print(alpha.shape)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

n_bins = 100

ax1.hist(alpha, bins=n_bins)
ax1.set_xlabel("alpha")

ax2.hist(kappa0, bins=n_bins)
ax2.set_xlabel("k0")

plt.tight_layout()
plt.show()
