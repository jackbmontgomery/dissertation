import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import EqualDiffusionReactionFDSolver
from src.params import EqualDiffusionReactionParams
from src.utils import adam_minimise, generate_noisy_samples
from src.voltammetry import CyclicDC

key = jr.key(0)
samples_key, key = jr.split(key, 2)

voltammetry = CyclicDC()
fd_solver = EqualDiffusionReactionFDSolver(voltammetry)
params = EqualDiffusionReactionParams(alpha=jnp.array(0.6), K0=jnp.array(10.0))
base_current = fd_solver.solve(params)
samples = generate_noisy_samples(10, base_current, 0.2, key=samples_key)


def log_likelihood_fn(params: EqualDiffusionReactionParams, samples=samples):
    current = fd_solver.solve(params)
    return -jnp.sum((samples - current) ** 2)


init_params = EqualDiffusionReactionParams(alpha=jnp.array(0.4), K0=jnp.array(5.0))
print(log_likelihood_fn(init_params))
params = adam_minimise(init_params, 1e-2, 1000, log_likelihood_fn)
print(log_likelihood_fn(params))

print(params.alpha, params.K0)
