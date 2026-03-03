from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import HeterogeneousReactionFDSolver
from src.params import (
    HeterogenousReactionParams,
)
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

from .base import AbstractSamplingExperiment


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
            dB=jnp.array(0.1),
            dC=jnp.array(0.2),
            dD=jnp.array(0.8),
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

        generate_key, sampling_key, key = jr.split(key, 3)

        fdm_solver = HeterogeneousReactionFDSolver(voltammetry)

        base_current = fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            noise,
            key=key,
        )

        def logdensity_fn(params: HeterogenousReactionParams, samples=samples):
            current = fdm_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = HeterogenousReactionParams(
            alpha_1=jnp.linspace(0.3, 0.7, NUM_CPUS),
            K0_1=jnp.linspace(5.0, 15.0, NUM_CPUS),
            Ef_1=jnp.linspace(0.2, 0.7, NUM_CPUS),
            alpha_2=jnp.linspace(0.3, 0.7, NUM_CPUS),
            K0_2=jnp.linspace(5.0, 15.0, NUM_CPUS),
            Ef_2=jnp.linspace(0.2, 0.7, NUM_CPUS),
            dB=jnp.linspace(0.1, 0.8, NUM_CPUS),
            dC=jnp.linspace(0.1, 0.8, NUM_CPUS),
            dD=jnp.linspace(0.1, 0.8, NUM_CPUS),
            K_het=jnp.linspace(5.0, 30.0, NUM_CPUS),
        )

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
            dB=samples.dB,
            dC=samples.dC,
            dD=samples.dD,
            K_het=samples.K_het,
            logdensity=logdensity,
        )

        print(f"Time Taken: {end_time - start_time:.2f}s")
        print(f"Number of Samples: {len(samples.alpha_1.flatten())}")
        print(f"Data Type: {samples.alpha_1.dtype}")

        for k, v in info.items():
            print(f"{k}: {v}")
