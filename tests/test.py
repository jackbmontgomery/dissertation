import inspect
import multiprocessing
from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from tabulate import tabulate

from src.fdm import ElectronReactionFDSolver
from src.optimisers import make_adam_optimise
from src.params import ElectronReactionParams
from src.reaction import ElectronReaction
from src.utils import generate_noisy_samples, pretty_header
from src.voltammetry import CyclicDC


def _property_names(params: ElectronReactionParams) -> list[str]:
    return [
        name
        for name, val in inspect.getmembers(
            type(params), lambda v: isinstance(v, property)
        )
    ]


voltammetry = CyclicDC()
reaction = ElectronReaction()
fd_solver = ElectronReactionFDSolver(voltammetry)
num_experimental_samples = 10
experimental_noise = 0.02
num_chains: int = multiprocessing.cpu_count()
optim_learning_rate = 1e-1
optim_steps = 25
warmup_step_size = 5e-1
seed = 0


print(pretty_header(reaction))

key = jr.key(seed)
key_samples, key_warmup, key_init = jr.split(key, 3)

_, base_current = fd_solver.solve(reaction.true_parameters)

experimental_samples = generate_noisy_samples(
    num_experimental_samples,
    base_current,
    experimental_noise,
    key=key_samples,
)


def logdensity_fn(params: ElectronReactionParams):
    _, current = fd_solver.solve(params)
    return -jnp.sum((experimental_samples - current) ** 2)


init_params = reaction.create_init_params(key_init, num_chains)

print(pretty_header("ADAM Optimisation", char="~"))
adam_start_time = perf_counter()

adam_minimise = make_adam_optimise(
    num_steps=optim_steps,
    log_density=logdensity_fn,
    learning_rate=optim_learning_rate,
)

optimised_parameters, log_densities, _ = vmap(adam_minimise)(init_params)
log_densities.block_until_ready()

print(f"Optimisation Time: {perf_counter() - adam_start_time:.2f}s")
print("Final Log Densities:")
print(log_densities[:, -1])

headers = _property_names(optimised_parameters)
rows = []
for i in range(num_chains):
    params = tree_util.tree_map(lambda x: x[i], optimised_parameters)
    rows.append([f"{getattr(params, name):.6f}" for name in headers])

print(
    tabulate(
        rows,
        headers=headers,
        tablefmt="fancy_grid",
        showindex=[f"Chain {i}" for i in range(num_chains)],
    )
)
