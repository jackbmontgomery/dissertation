import jax.numpy as jnp
import jax.random as jr
from chex import PRNGKey

from src.params import HeterogenousReactionParams

from ._base import AbstractReaction


class HeterogeneousReaction(AbstractReaction):
    @property
    def true_parameters(self) -> HeterogenousReactionParams:
        return HeterogenousReactionParams(
            alpha_1=jnp.array(0.6),
            K0_1=jnp.array(10.0),
            Ef_1=jnp.array(0.5),
            alpha_2=jnp.array(0.4),
            K0_2=jnp.array(5.0),
            Ef_2=jnp.array(0.2),
            K_het=jnp.array(20.0),
        )

    def __str__(self) -> str:
        return "HeterogeneousReaction"

    def create_init_params(self, key: PRNGKey, num: int):
        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

        alpha_1_vals = jnp.linspace(0.5, 0.7, num)
        K0_1_vals = jnp.linspace(1.0, 20.0, num)
        Ef_1_vals = jnp.linspace(0.0, 1.0, num)
        alpha_2_vals = jnp.linspace(0.3, 0.5, num)
        K0_2_vals = jnp.linspace(1.0, 10.0, num)
        Ef_2_vals = jnp.linspace(0.0, 1.0, num)
        K_het_vals = jnp.linspace(5.0, 30.0, num)

        alpha_1 = jr.permutation(k1, alpha_1_vals)
        K0_1 = jr.permutation(k2, K0_1_vals)
        Ef_1 = jr.permutation(k3, Ef_1_vals)
        alpha_2 = jr.permutation(k4, alpha_2_vals)
        K0_2 = jr.permutation(k5, K0_2_vals)
        Ef_2 = jr.permutation(k6, Ef_2_vals)
        K_het = jr.permutation(k7, K_het_vals)

        return HeterogenousReactionParams(
            alpha_1=alpha_1,
            K0_1=K0_1,
            Ef_1=Ef_1,
            alpha_2=alpha_2,
            K0_2=K0_2,
            Ef_2=Ef_2,
            K_het=K_het,
        )
