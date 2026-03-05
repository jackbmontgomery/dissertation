from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from chex import PRNGKey

from src.fdm import HeterogeneousReactionFDSolver
from src.params import HeterogenousReactionParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

from .base import AbstractSamplingExperiment


def create_init_params(key: PRNGKey, num_chains: int):
    k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

    alpha_1_vals = jnp.linspace(0.5, 0.7, num_chains)
    K0_1_vals = jnp.linspace(1.0, 20.0, num_chains)
    Ef_1_vals = jnp.linspace(0.0, 1.0, num_chains)
    alpha_2_vals = jnp.linspace(0.3, 0.5, num_chains)
    K0_2_vals = jnp.linspace(1.0, 20.0, num_chains)
    Ef_2_vals = jnp.linspace(0.0, 1.0, num_chains)
    K_het_vals = jnp.linspace(5.0, 30.0, num_chains)

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


class HeterogeneousReactionSamplingExperiment(AbstractSamplingExperiment):
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

    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        noise: float,
        sigma: int,
        seed: int = 0,
    ):
        key = jr.key(seed)
        voltammetry = CyclicDC(sigma=sigma)

        param_key, sampling_key, key = jr.split(key, 3)

        fdm_solver = HeterogeneousReactionFDSolver(voltammetry)

        _, base_current = fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            noise,
            key=key,
        )

        def logdensity_fn(params: HeterogenousReactionParams, samples=samples):
            _, current = fdm_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = create_init_params(param_key, NUM_CPUS)

        start_time = perf_counter()
        samples, logdensity, info = sampling_algorithm(
            sampling_key, init_params, logdensity_fn
        )
        samples.alpha_1.block_until_ready()
        end_time = perf_counter()

        data_file = f"H_{sampling_algorithm}_{noise:.2f}_{sigma:.0f}.npz"

        np.savez_compressed(
            f"./data/{data_file}",
            alpha_1=samples.alpha_1,
            K0_1=samples.K0_1,
            Ef_1=samples.Ef_1,
            alpha_2=samples.alpha_2,
            K0_2=samples.K0_2,
            Ef_2=samples.Ef_2,
            K_het=samples.K_het,
            logdensity=logdensity,
        )

        print(f"Time Taken: {end_time - start_time:.2f}s")

        for k, v in info.items():
            print(f"{k}: {v}")
