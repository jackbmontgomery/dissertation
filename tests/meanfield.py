import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import optimistix as optx
from jax import jit, vmap
from jax.nn import sigmoid
from jax.scipy.special import logit
from tqdm import tqdm

from src.voltammetry import LinearSweepDC
from src.fdm_discretisation import ButlerVolmerFDMDiscretisation1D, uniform_discretise
from src.pde_parameters import ButlerVolmerInverseParameters, bv_inverse_to_physical
from src.sampling import generate_noisy_samples
from src.simulate import create_fdm_current_simulator

key = jr.key(0)

true_params = ButlerVolmerInverseParameters(a=logit(0.6), k0=jnp.log(100.0))

phy_params = bv_inverse_to_physical(true_params)

experiment = LinearSweepDC()

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


bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
init_params = ButlerVolmerInverseParameters(a=logit(0.1), k0=jnp.log(10.0))
solution = optx.minimise(lambda params, _: -log_density(params), bfgs, init_params)
sampling_init_params = solution.value

print(sigmoid(sampling_init_params.a), jnp.exp(sampling_init_params.k0))

approx_key, sample_key, key = jr.split(key, 3)
optimiser = optax.adam(5e-1)
mf = blackjax.meanfield_vi(log_density, optimiser)
state = mf.init(true_params)

step_key, key = jr.split(key, 2)
state, info = mf.step(step_key, state)
print(info)
jit_step = jit(mf.step)
for i in tqdm(range(500)):
    step_key, key = jr.split(key, 2)
    state, info = jit_step(step_key, state)

print(info)

sample_key, key = jr.split(key, 2)
samples = mf.sample(sample_key, state, 10)

# print(sigmoid(state.position.a))
# print(jnp.exp(state.position.k0))

# samples, _ = pathfinder.sample(sample_key, state, 5000)

alpha = sigmoid(samples.a)
kappa0 = jnp.exp(samples.k0)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
n_bins = 50
ax1.hist(alpha, bins=n_bins)
ax2.hist(kappa0, bins=n_bins)
plt.show()
# np.savez_compressed("./data/vi_dc.npz", alpha=alpha, kappa0=kappa0)
