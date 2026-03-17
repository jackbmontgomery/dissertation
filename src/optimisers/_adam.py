from typing import Callable, Tuple

import optax
from chex import dataclass
from equinox import apply_updates, filter, filter_value_and_grad, is_array
from jax.lax import scan
from jaxtyping import PyTree, Scalar

from src.params import Params


@dataclass
class AdamMinimiseCarry:
    opt_state: PyTree
    params: Params


def make_adam_optimise(
    num_steps: int, log_density: Callable, *, learning_rate: float
) -> Callable[[Params], Tuple[Params, Scalar, Params]]:
    optim = optax.adam(learning_rate)

    def _adam_optimise(init_params: Params) -> Tuple[Scalar, Params]:
        value_grad_fn = filter_value_and_grad(lambda x: -log_density(x))

        def step_fn(carry: AdamMinimiseCarry, _):
            params, opt_state = carry.params, carry.opt_state
            log_density, grads = value_grad_fn(params)
            updates, new_opt_state = optim.update(
                grads, opt_state, filter(params, is_array)
            )
            new_params = apply_updates(params, updates)
            new_carry = AdamMinimiseCarry(opt_state=new_opt_state, params=new_params)
            return new_carry, (log_density, params)

        opt_state = optim.init(filter(init_params, is_array))

        init_carry = AdamMinimiseCarry(opt_state=opt_state, params=init_params)
        final_carry, (log_densities, params_path) = scan(
            step_fn, init_carry, None, num_steps
        )

        return final_carry.params, -1 * log_densities, params_path

    return _adam_optimise
