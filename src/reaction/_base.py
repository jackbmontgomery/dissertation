from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from chex import PRNGKey

from src.params import Params


class AbstractReaction(ABC):
    @property
    @abstractmethod
    def true_parameters(self) -> Params:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_init_params(self, key: PRNGKey, num: int) -> Params:
        raise NotImplementedError


def latin_hypercube(key: PRNGKey, num_samples: int, num_dims: int):
    keys = jr.split(key, num_dims)
    strata = []
    for i in range(num_dims):
        k1, k2 = jr.split(keys[i])
        u = jr.uniform(k1, (num_samples,))
        perm = jr.permutation(k2, num_samples)
        strata.append((perm + u) / num_samples)
    return jnp.stack(strata, axis=-1)
