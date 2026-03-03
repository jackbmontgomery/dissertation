import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import EqualDiffusionReactionFDSolver
from src.params import EqualDiffusionReactionParams
from src.utils import generate_noisy_samples
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


alpha_params = jnp.linspace(0.5, 0.7, 100)
K0_params = jnp.linspace(5.0, 15.0, 100)

alpha_grid, K0_grid = jnp.meshgrid(alpha_params, K0_params, indexing="ij")

grid_params = EqualDiffusionReactionParams(
    alpha=alpha_grid.flatten(), K0=K0_grid.flatten()
)

likelihood_grid = vmap(log_likelihood_fn)(grid_params).reshape(alpha_grid.shape)

# plt.figure()
# plt.pcolormesh(alpha_grid, K0_grid, likelihood_grid, shading="auto")
# plt.xlabel("alpha")
# plt.ylabel("K0")
# plt.colorbar(label="log-likelihood")
# plt.show()

plt.figure()
plt.contourf(
    alpha_grid,
    K0_grid,
    likelihood_grid,
    levels=30,
)
plt.xlabel("alpha")
plt.ylabel("K0")
plt.colorbar(label="log-likelihood")
plt.show()
