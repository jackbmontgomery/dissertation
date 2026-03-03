from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import optax
import optimistix as optx
from chex import dataclass
from equinox import apply_updates, filter, filter_grad, is_array
from jax import vmap
from jax.lax import scan
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from src.params import Params


@dataclass
class AdamMinimiseCarry:
    opt_state: PyTree
    params: Params


def generate_noisy_samples(
    num_samples: int,
    current: Scalar,
    sigma: float,
    *,
    key: PRNGKeyArray,
):
    def add_noise(r_key: PRNGKeyArray, current=current):
        noisy_current = current + jr.normal(r_key, shape=current.shape) * sigma
        return noisy_current

    keys = jr.split(key, num_samples)
    samples = vmap(add_noise)(keys)
    return samples


def interleave_concat_2d(a, b):
    return jnp.column_stack([a, b]).reshape(-1)


def interleave_concat_4d(a, b, c, d):
    return jnp.column_stack([a, b, c, d]).reshape(-1)


def adam_minimise(
    params: Params, learning_rate: float, num_steps: int, log_density: Callable
) -> Params:
    optim = optax.adam(learning_rate)
    grad_fn = filter_grad(lambda x: -log_density(x))

    def step_fn(carry: AdamMinimiseCarry, _):
        params, opt_state = carry.params, carry.opt_state
        grads = grad_fn(params)
        updates, new_opt_state = optim.update(
            grads, opt_state, filter(params, is_array)
        )
        new_params = apply_updates(params, updates)
        new_carry = AdamMinimiseCarry(opt_state=new_opt_state, params=new_params)
        return new_carry, None

    opt_state = optim.init(filter(params, is_array))

    init_carry = AdamMinimiseCarry(opt_state=opt_state, params=params)
    final_carry, _ = scan(step_fn, init_carry, None, num_steps)
    return final_carry.params


def bfgs_minimise(initial_parameters, log_density: Callable):
    bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
    solution = optx.minimise(
        lambda params, _: -log_density(params), bfgs, initial_parameters
    )
    sampling_init_params = solution.value
    return sampling_init_params
