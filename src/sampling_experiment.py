import gzip
import multiprocessing
import pickle
from time import perf_counter
from typing import Dict

import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from tabulate import tabulate

from src.fdm import AbstractFDSolver
from src.optimisers import make_adam_optimise
from src.params import Params, param_property_names
from src.reaction import AbstractReaction
from src.sampling import NUTSSampler, RWMHSampler
from src.utils import generate_noisy_samples, pretty_header


def print_infos(info: Dict):
    for key, item in info.items():
        print(key, item)


def print_optim(params, log_densities, num_chains):
    property_names = param_property_names(params)
    display_names = [name for name in property_names if name != "theta_sol"]

    rows = []
    for i in range(num_chains):
        chain_params = tree_util.tree_map(lambda x: x[i], params)
        row_data = [f"{getattr(chain_params, name):.2f}" for name in display_names]
        row_data.append(f"{log_densities[i, -1]:.2f}")
        rows.append(row_data)

    headers = display_names + ["Log Density"]

    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="fancy_grid",
            showindex=[f"Chain {i}" for i in range(num_chains)],
        )
    )


def sampling_experiment(
    reaction: AbstractReaction,
    sampling_fd_solver: AbstractFDSolver,
    *,
    data_fd_solver: AbstractFDSolver | None = None,
    num_experimental_samples: int = 10,
    experimental_noise: float = 0.02,
    num_chains: int = multiprocessing.cpu_count(),
    optim_learning_rate: float = 1e-1,
    optim_steps: int = 250,
    warmup_step_size: float = 5e-1,
    num_rwmh_samples: int,
    num_nuts_samples: int,
    seed: int,
    save: bool,
):
    if data_fd_solver is None:
        data_fd_solver = sampling_fd_solver

    print(pretty_header(reaction))

    key = jr.key(seed)
    key_samples, key_warmup, key_init = jr.split(key, 3)

    base_current = data_fd_solver.solve(reaction.true_parameters)

    experimental_samples = generate_noisy_samples(
        num_experimental_samples,
        base_current,
        experimental_noise,
        key=key_samples,
    )

    def logdensity_fn(params: Params):
        current = sampling_fd_solver.solve(params)
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
    print_optim(optimised_parameters, log_densities, num_chains)

    best_idx = jnp.argmax(log_densities[:, -1])
    best_params = jtu.tree_map(lambda x: x[best_idx], optimised_parameters)

    print(pretty_header("Window Adaption", char="~"))

    adaption_start_time = perf_counter()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        initial_step_size=warmup_step_size,
    )

    (last_states, window_adaption_params), _ = warmup.run(key_warmup, best_params)

    flat_last_states, _ = ravel_pytree(last_states)
    flat_last_states.block_until_ready()

    print(f"Adaption Time: {perf_counter() - adaption_start_time:.2f}s")
    print(f"Step size: {window_adaption_params['step_size']:.2f}")

    print(pretty_header("RWMH", char="-"))
    rwmh_start_time = perf_counter()

    # Roberts, Gelman & Gilks (1997): optimal RWMH proposal covariance
    # is (2.38^2 / D) * Sigma_target, targeting ~23.4% acceptance rate

    scale = (2.38**2) / reaction.parameter_dim
    proposal_cov = scale * window_adaption_params["inverse_mass_matrix"]
    sigma_rwmh = jnp.linalg.cholesky(proposal_cov)

    key, key_rwmh = jr.split(key)

    rwmh = RWMHSampler(logdensity_fn, num_rwmh_samples, num_chains)

    rwmh_params = {"random_step": blackjax.mcmc.random_walk.normal(sigma_rwmh)}

    rwmh_samples, infos = rwmh.run(optimised_parameters, rwmh_params, key=key_rwmh)

    flat_rwmh_samples, _ = ravel_pytree(rwmh_samples)
    flat_rwmh_samples.block_until_ready()

    print(f"Sampling Time: {perf_counter() - rwmh_start_time:.2f}s")
    print_infos(infos)

    print(pretty_header("NUTS", char="-"))

    key, key_nuts = jr.split(key)
    nuts = NUTSSampler(logdensity_fn, num_nuts_samples, num_chains)

    nuts_start_time = perf_counter()

    nuts_samples, infos = nuts.run(
        optimised_parameters, window_adaption_params, key=key_nuts
    )

    flat_nuts_samples, _ = ravel_pytree(rwmh_samples)
    flat_nuts_samples.block_until_ready()

    print(f"Sampling Time: {perf_counter() - nuts_start_time:.2f}s")
    print_infos(infos)

    if save:
        file_name = f"reaction={reaction},noise={experimental_noise},seed={seed}"

        with gzip.open(f"./data/sampling/{file_name}.pkl.gz", "wb") as f:
            pickle.dump({"nuts": nuts_samples, "rwmh": rwmh_samples}, f)
