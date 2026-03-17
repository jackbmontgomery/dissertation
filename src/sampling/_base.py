from abc import ABC, abstractmethod
from typing import Callable, Dict, NamedTuple, Tuple

import jax.random as jr
from jax import jit, pmap
from jax.lax import scan
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from src.params import Params

LogDensity = Callable[[PyTree], Scalar]


class AbstractSampler(ABC):
    @abstractmethod
    def warmup(
        self, params: Params, learning_rate: float, steps: int
    ) -> Tuple[NamedTuple, Dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, init_states: NamedTuple, n_samples: int) -> Tuple[Params, Dict]:
        raise NotImplementedError


def inference_loop(
    key: PRNGKeyArray,
    kernel: Callable[[PRNGKeyArray, PyTree], Tuple[PyTree, PyTree]],
    initial_state: Params,
    num_samples: int,
):
    @jit
    def scan_step(state, step_key):
        state, info = kernel(step_key, state)
        return state, (state, info)

    keys = jr.split(key, num_samples)
    _, (states, infos) = scan(scan_step, initial_state, keys)

    return states, infos


inference_loop_multiple_chains: Callable = pmap(
    inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
)
