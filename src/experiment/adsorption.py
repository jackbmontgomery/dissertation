import jax.numpy as jnp
import jax.random as jr
import numpy as np
from chex import PRNGKey

from src.fdm import AdsorptionReactionExplicitFDSolver, AdsorptionReactionNewtonFDSolver
from src.params import AdsorptionReactionParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

from .base import AbstractSamplingExperiment


def create_init_params(key: PRNGKey, num_chains: int):
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)

    alpha_sol_vals = jnp.linspace(0.3, 0.5, num_chains)
    K0_sol_vals = jnp.linspace(1e-6, 5.0, num_chains)
    Ef_sol_vals = jnp.linspace(-0.5, 0.5, num_chains)
    alpha_ads_vals = jnp.linspace(0.3, 0.5, num_chains)
    K0_ads_vals = jnp.linspace(1e-6, 5.0, num_chains)
    K_A_ads_vals = jnp.linspace(1e-6, 5.0, num_chains)
    K_A_des_vals = jnp.linspace(1e-6, 5.0, num_chains)
    K_B_ads_vals = jnp.linspace(1e-6, 5.0, num_chains)
    K_B_des_vals = jnp.linspace(1e-6, 5.0, num_chains)

    alpha_sol = jr.permutation(k1, alpha_sol_vals)
    K0_sol = jr.permutation(k2, K0_sol_vals)
    Ef_sol = jr.permutation(k3, Ef_sol_vals)
    alpha_ads = jr.permutation(k4, alpha_ads_vals)
    K0_ads = jr.permutation(k5, K0_ads_vals)
    K_A_ads = jr.permutation(k6, K_A_ads_vals)
    K_A_des = jr.permutation(k7, K_A_des_vals)
    K_B_ads = jr.permutation(k8, K_B_ads_vals)
    K_B_des = jr.permutation(k9, K_B_des_vals)

    return AdsorptionReactionParams(
        alpha_sol=alpha_sol,
        K0_sol=K0_sol,
        Ef_sol=Ef_sol,
        alpha_ads=alpha_ads,
        K0_ads=K0_ads,
        K_A_ads=K_A_ads,
        K_A_des=K_A_des,
        K_B_ads=K_B_ads,
        K_B_des=K_B_des,
    )


class AdsorptionSamplingExperiment(AbstractSamplingExperiment):
    @property
    def true_parameters(self) -> AdsorptionReactionParams:
        return AdsorptionReactionParams(
            alpha_sol=jnp.array(0.4),
            K0_sol=jnp.array(1e-3),
            Ef_sol=jnp.array(0.0),
            alpha_ads=jnp.array(0.45),
            K0_ads=jnp.array(5e-1),
            K_A_ads=jnp.array(4.5),
            K_A_des=jnp.array(1.0),
            K_B_ads=jnp.array(1.0),
            K_B_des=jnp.array(1.0),
        )

    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        noise: float,
        sigma: int,
        seed: int = 0,
    ):
        key = jr.key(seed)
        voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=sigma)

        param_key, sampling_key, key = jr.split(key, 3)

        exp_fdm_solver = AdsorptionReactionNewtonFDSolver(voltammetry)

        _, base_current = exp_fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            noise,
            key=key,
        )

        fd_solver = AdsorptionReactionExplicitFDSolver(voltammetry)

        def logdensity_fn(params: AdsorptionReactionParams, samples=samples):
            _, current = fd_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = create_init_params(param_key, NUM_CPUS)

        samples, logdensity, info = sampling_algorithm(
            sampling_key, init_params, logdensity_fn
        )

        data_file = f"A_{sampling_algorithm}_{noise:.2f}_{sigma:.0f}.npz"

        np.savez_compressed(
            f"./data/{data_file}",
            alpha_sol=samples.alpha_sol,
            K0_sol=samples.K0_sol,
            Ef_sol=samples.Ef_sol,
            alpha_ads=samples.alpha_ads,
            K0_ads=samples.K0_ads,
            K_A_ads=samples.K_A_ads,
            K_A_des=samples.K_A_des,
            K_B_ads=samples.K_B_ads,
            K_B_des=samples.K_B_des,
            logdensity=logdensity,
        )
