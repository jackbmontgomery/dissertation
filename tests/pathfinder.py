from time import perf_counter

import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from jax import checkpoint, vmap
from jax.nn import sigmoid
from jax.scipy.special import logit

from src.voltammetry import LinearSweepAC, LinearSweepDC
from src.fdm_discretisation import ButlerVolmerFDMDiscretisation1D, uniform_discretise
from src.pde_parameters import ButlerVolmerInverseParameters, bv_inverse_to_physical
from src.sampling import generate_noisy_samples
from src.simulate import create_fdm_current_simulator

key = jr.key(0)

true_params = ButlerVolmerInverseParameters(
    a=logit(0.6), k0=jnp.log(100.0), e0=5.0 * jnp.arctanh(2.0 / 10.0)
)

phy_params = bv_inverse_to_physical(true_params)

experiment = LinearSweepAC()

T, X = uniform_discretise(experiment)
print(f"T:{T.shape},X:{X.shape}")

potentials = vmap(experiment.potential)(T)

fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, T)
c_init = jnp.ones_like(X)

simulate_current = create_fdm_current_simulator(
    c_init, potentials, fdm_discretisation, X
)

samples = generate_noisy_samples(10, simulate_current, phy_params, 0.1, key=key)


def log_density(params: ButlerVolmerInverseParameters, samples=samples):
    phy_params = bv_inverse_to_physical(params)
    pred = simulate_current(phy_params)
    return -jnp.sum((samples - pred) ** 2)


init_params = ButlerVolmerInverseParameters(
    a=logit(0.3), k0=jnp.log(80.0), e0=jnp.arctanh(0.0)
)

approx_key, sample_key, key = jr.split(key, 3)
pathfinder = blackjax.pathfinder(log_density)
start = perf_counter()
print("Starting...")
state, info = pathfinder.approximate(approx_key, init_params, ftol=1e-8)
print("Elbo Path:", info.path.elbo)

print("Apporx Params", sigmoid(state.position.a), jnp.exp(state.position.k0))
samples, _ = pathfinder.sample(sample_key, state, 40_000)
end = perf_counter()
print("Finished... Time taken", end - start)

alpha = sigmoid(samples.a)
kappa0 = jnp.exp(samples.k0)
eps0 = 10.0 * jnp.tanh(samples.e0 / 5.0)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
n_bins = 50
ax1.hist(alpha, bins=n_bins)
ax2.hist(kappa0, bins=n_bins)
plt.show()
np.savez_compressed("./data/vi_ac.npz", alpha=alpha, kappa0=kappa0, eps0=eps0)
