from typing import Callable, Tuple

import jax
import jax.random as jr
from evosax.algorithms import CMA_ES
from jax.flatten_util import ravel_pytree
from jax.lax import scan
from jaxtyping import PRNGKeyArray, Scalar

from src.params import Params


def make_cmaes_optimise(
    num_iterations: int, log_density: Callable, *, population_size: int = 32
) -> Callable[[Params, PRNGKeyArray], Tuple[Params, Scalar, Params]]:
    def _cmaes_optimise(
        init_params: Params,
        key: PRNGKeyArray,
    ) -> Tuple[Scalar, Params]:
        fitness_fn = jax.vmap(lambda x: -log_density(x))

        es = CMA_ES(population_size=population_size, solution=init_params)
        es_params = es.default_params
        _, unflatten_fn = ravel_pytree(init_params)

        def step_fn(state, keys):
            key_ask, key_eval = keys
            population, state = es.ask(key_ask, state, es_params)
            fitness = fitness_fn(population)
            state, metrics = es.tell(key_eval, population, fitness, state, es_params)
            return state, (state.best_fitness, unflatten_fn(state.best_solution))

        init_state = es.init(key, init_params, es_params)

        keys = jr.split(key, num_iterations * 2).reshape(num_iterations, 2)
        final_state, (log_densities, params_path) = scan(step_fn, init_state, keys)

        return final_state.best_solution, -1 * log_densities, params_path

    return _cmaes_optimise
