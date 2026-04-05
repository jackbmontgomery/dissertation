import jax.numpy as jnp
from chex import PRNGKey

from src.params import HeterogenousReactionParams

from ._base import AbstractReaction, latin_hypercube


class HeterogeneousReaction(AbstractReaction):
    @property
    def true_parameters(self) -> HeterogenousReactionParams:
        return HeterogenousReactionParams(
            alpha_1=jnp.array(0.5),
            K0_1=jnp.array(15.0),
            thetaf_1=jnp.array(0.2),
            alpha_2=jnp.array(0.5),
            K0_2=jnp.array(6.0),
            thetaf_2=jnp.array(0.4),
            K_het=jnp.array(25.0),
        )

    @property
    def parameter_dim(self) -> int:
        return 7

    def __str__(self) -> str:
        return "HeterogeneousReaction"

    def create_init_params(self, key: PRNGKey, num: int, scale: float = 2.0):
        true_p = self.true_parameters
        centre = jnp.array(
            [
                true_p._alpha_1_inv,
                true_p._K0_1_inv,
                true_p._thetaf_1_inv,
                true_p._alpha_2_inv,
                true_p._K0_2_inv,
                true_p._thetaf_2_inv,
                true_p._K_het_inv,
            ]
        )

        unit = latin_hypercube(key, num, num_dims=7)
        samples = centre + scale * (2 * unit - 1)

        return HeterogenousReactionParams.from_transformed(
            alpha_1_inv=samples[:, 0],
            K0_1_inv=samples[:, 1],
            thetaf_1_inv=samples[:, 2],
            alpha_2_inv=samples[:, 3],
            K0_2_inv=samples[:, 4],
            thetaf_2_inv=samples[:, 5],
            K_het_inv=samples[:, 6],
        )
