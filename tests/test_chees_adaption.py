import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.reaction import ElectronReaction
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

seed = 0
num_chains = 8

reaction = ElectronReaction()
voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)

key = jr.key(seed)
key_samples, key_init = jr.split(key, 2)

_, base_current = fd_solver.solve(reaction.true_parameters)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_samples,
)


def logdensity_fn(params: ElectronReactionParams):
    _, current = fd_solver.solve(params)
    return -jnp.sum((experimental_samples - current) ** 2)


init_params = reaction.create_init_params(key_init, num_chains)

warmup = blackjax.chees_adaptation(logdensity_fn, num_chains, max_leapfrog_steps=200)

optim = optax.adam(1e-2)

(initial_states, hmc_params), warmup_info = warmup.run(
    key,
    init_params,
    step_size=1e-3,
    optim=optim,
    num_steps=200,
)
print(initial_states)
print("-------")
print(hmc_params)
print("-------")

positions = warmup_info.state.position
leaves = jtu.tree_leaves(positions)
stacked = jnp.stack(leaves, axis=-1)

all_positions = stacked[100:].reshape(-1, 3)
Sigma_hat = jnp.cov(all_positions.T)

D = 3
scale = (2.38**2) / D
sigma_rwmh = scale * Sigma_hat

D = 3
scale = (2.38**2) / D
sigma_rwmh = scale * Sigma_hat
print(Sigma_hat)
