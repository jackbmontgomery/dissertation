from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from chex import Scalar, dataclass
from equinox import apply_updates, filter, filter_value_and_grad, is_array
from jax.lax import scan
from jaxtyping import PyTree

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

key = jr.key(0)
cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)

true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.0),
)
init_params = ElectronReactionParams(
    alpha=jnp.array(0.5),
    K0=jnp.array(15.0),
    Ef=jnp.array(0.5),
)

_, base_current = fd_solver.solve(true_params)
experimental_samples = generate_noisy_samples(10, base_current, 0.25, key=key)


def log_density(params: ElectronReactionParams, samples=experimental_samples):
    _, current = fd_solver.solve(params)
    return -jnp.sum((samples - current) ** 2)


@dataclass
class AdamMinimiseCarry:
    opt_state: PyTree
    params: ElectronReactionParams


def adam_minimise(
    params: ElectronReactionParams,
    learning_rate: float,
    num_steps: int,
    log_density: Callable,
) -> Tuple[ElectronReactionParams, Scalar, ElectronReactionParams]:
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
