import jax.numpy as jnp
import jax.random as jr
from chex import PRNGKey

from src.params import ElectronReactionParams

from ._base import AbstractReaction


class ElectronReaction(AbstractReaction):
    @property
    def true_parameters(self) -> ElectronReactionParams:
        return ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(10.0),
            Ef=jnp.array(0.5),
        )

    def __str__(self) -> str:
        return "ElectronReaction"

    def create_init_params(self, key: PRNGKey, num: int):
        k1, k2, k3 = jr.split(key, 3)

        alpha_vals = jnp.linspace(0.5, 0.7, num)
        K0_vals = jnp.linspace(5.0, 20.0, num)
        Ef_vals = jnp.linspace(0.0, 1.0, num)

        alpha = jr.permutation(k1, alpha_vals)
        K0 = jr.permutation(k2, K0_vals)
        Ef = jr.permutation(k3, Ef_vals)

        return ElectronReactionParams(
            alpha=alpha,
            K0=K0,
            Ef=Ef,
        )
