import jax.numpy as jnp
from chex import PRNGKey

from src.params import AdsorptionReactionParams

from ._base import AbstractReaction, latin_hypercube


class AdsorptionReaction(AbstractReaction):
    @property
    def true_parameters(self) -> AdsorptionReactionParams:
        return AdsorptionReactionParams(
            alpha_sol=jnp.array(0.5),
            K0_sol=jnp.array(1e-3),
            thetaf_sol=jnp.array(0.0),
            alpha_ads=jnp.array(0.45),
            K0_ads=jnp.array(5e-1),
            K_A_ads=jnp.array(4.5),
            K_A_des=jnp.array(1.0),
            K_B_ads=jnp.array(1.0),
            K_B_des=jnp.array(1.0),
        )

    @property
    def parameter_dim(self) -> int:
        return 9

    def __str__(self) -> str:
        return "AdsorptionReaction"

    def create_init_params(self, key: PRNGKey, num: int, scale: float = 2.0):
        true_p = self.true_parameters
        centre = jnp.array(
            [
                true_p._alpha_sol_inv,
                true_p._K0_sol_inv,
                true_p._thetaf_sol_inv,
                true_p._alpha_ads_inv,
                true_p._K0_ads_inv,
                true_p._K_A_ads_inv,
                true_p._K_A_des_inv,
                true_p._K_B_ads_inv,
                true_p._K_B_des_inv,
            ]
        )
        unit = latin_hypercube(key, num, num_dims=9)
        samples = centre + scale * (2 * unit - 1)

        return AdsorptionReactionParams.from_transformed(
            alpha_sol_inv=samples[:, 0],
            K0_sol_inv=samples[:, 1],
            thetaf_sol_inv=samples[:, 2],
            alpha_ads_inv=samples[:, 3],
            K0_ads_inv=samples[:, 4],
            K_A_ads_inv=samples[:, 5],
            K_A_des_inv=samples[:, 6],
            K_B_ads_inv=samples[:, 7],
            K_B_des_inv=samples[:, 8],
        )
