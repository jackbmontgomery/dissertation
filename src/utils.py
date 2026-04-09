from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as jr
import optax
from chex import dataclass
from equinox import apply_updates, filter, filter_value_and_grad, is_array
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
    percentage_noise: float,
    *,
    key: PRNGKeyArray,
):
    max_current = jnp.min(current)
    sigma = percentage_noise * max_current

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
) -> Tuple[Params, Scalar, Params]:
    optim = optax.adam(learning_rate)
    value_grad_fn = filter_value_and_grad(lambda x: -log_density(x))

    def step_fn(carry: AdamMinimiseCarry, _):
        params, opt_state = carry.params, carry.opt_state
        log_likelihood, grads = value_grad_fn(params)
        updates, new_opt_state = optim.update(
            grads, opt_state, filter(params, is_array)
        )
        new_params = apply_updates(params, updates)
        new_carry = AdamMinimiseCarry(opt_state=new_opt_state, params=new_params)
        return new_carry, (-log_likelihood, params)

    opt_state = optim.init(filter(params, is_array))

    init_carry = AdamMinimiseCarry(opt_state=opt_state, params=params)
    final_carry, (log_likelihood, params_path) = scan(
        step_fn, init_carry, None, num_steps
    )
    return final_carry.params, log_likelihood, params_path


def pretty_header(text, width=60, char="="):
    return f"{f' {text} ':{char}^{width}}"
