import jax.numpy as jnp
import jax.random as jr
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

    def create_init_params(self, key: PRNGKey, num: int):
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)

        alpha_sol_vals = jnp.linspace(0.3, 0.5, num)
        K0_sol_vals = jnp.linspace(1e-6, 5.0, num)
        thetaf_sol_vals = jnp.linspace(-0.5, 0.5, num)
        alpha_ads_vals = jnp.linspace(0.3, 0.5, num)
        K0_ads_vals = jnp.linspace(1e-6, 5.0, num)
        K_A_ads_vals = jnp.linspace(1e-6, 5.0, num)
        K_A_des_vals = jnp.linspace(1e-6, 5.0, num)
        K_B_ads_vals = jnp.linspace(1e-6, 5.0, num)
        K_B_des_vals = jnp.linspace(1e-6, 5.0, num)

        alpha_sol = jr.permutation(k1, alpha_sol_vals)
        K0_sol = jr.permutation(k2, K0_sol_vals)
        thetaf_sol = jr.permutation(k3, thetaf_sol_vals)
        alpha_ads = jr.permutation(k4, alpha_ads_vals)
        K0_ads = jr.permutation(k5, K0_ads_vals)
        K_A_ads = jr.permutation(k6, K_A_ads_vals)
        K_A_des = jr.permutation(k7, K_A_des_vals)
        K_B_ads = jr.permutation(k8, K_B_ads_vals)
        K_B_des = jr.permutation(k9, K_B_des_vals)

        return AdsorptionReactionParams(
            alpha_sol=alpha_sol,
            K0_sol=K0_sol,
            thetaf_sol=thetaf_sol,
            alpha_ads=alpha_ads,
            K0_ads=K0_ads,
            K_A_ads=K_A_ads,
            K_A_des=K_A_des,
            K_B_ads=K_B_ads,
            K_B_des=K_B_des,
        )

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
