from typing import Callable

import blackjax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Scalar


def ess_over_time(samples: Array, num_points: int) -> Scalar:
    chain_len = samples.shape[1]
    end_indices = jnp.linspace(
        chain_len / num_points, chain_len, num_points, dtype=jnp.int32
    )

    ess_fn: Callable = jit(blackjax.diagnostics.effective_sample_size)

    result = jnp.zeros(num_points)
    for i, idx in enumerate(end_indices):
        result = result.at[i].set(ess_fn(samples[:, :idx]))

    return result


def potential_scale_reduction_over_time(samples: Array, num_points: int) -> Scalar:
    chain_len = samples.shape[1]
    end_indices = jnp.linspace(
        chain_len / num_points, chain_len, num_points, dtype=jnp.int32
    )

    rhat_fn: Callable = jit(blackjax.diagnostics.potential_scale_reduction)

    result = jnp.zeros(num_points)
    for i, idx in enumerate(end_indices):
        result = result.at[i].set(rhat_fn(samples[:, :idx]))

    return result
