import jax.numpy as jnp
from chex import PRNGKey

from src.params import ElectronReactionParams

from ._base import AbstractReaction, latin_hypercube


class ElectronReaction(AbstractReaction):
    @property
    def true_parameters(self) -> ElectronReactionParams:
        return ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(10.0),
            thetaf=jnp.array(0.5),
        )

    @property
    def parameter_dim(self) -> int:
        return 3

    def __str__(self) -> str:
        return "ElectronReaction"

    def create_init_params(self, key: PRNGKey, num: int, scale: float = 2.0):
        true_p = self.true_parameters
        centre = jnp.array([true_p._alpha_inv, true_p._K0_inv, true_p._thetaf_inv])

        unit = latin_hypercube(key, num, num_dims=3)
        samples = centre + scale * (2 * unit - 1)

        return ElectronReactionParams.from_transformed(
            alpha_inv=samples[:, 0],
            K0_inv=samples[:, 1],
            thetaf_inv=samples[:, 2],
        )


class ReversibleElectronReaction(ElectronReaction):
    @property
    def true_parameters(self) -> ElectronReactionParams:
        return ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(1000.0),
            thetaf=jnp.array(0.5),
        )
