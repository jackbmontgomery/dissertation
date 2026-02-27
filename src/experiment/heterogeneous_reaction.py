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
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractSamplingExperiment


class HeterogeneousReactionSamplingExperiment(AbstractSamplingExperiment):
    @property
    def true_parameters(self) -> HeterogenousReactionParams:
        return HeterogenousReactionParams(
            alpha1=jnp.array(0.6),
            K1_0=jnp.array(10.0),
            E1_f=jnp.array(0.5),
            alpha2=jnp.array(0.6),
            K2_0=jnp.array(30.0),
            E2_f=jnp.array(0.2),
            dB=jnp.array(0.8),
            dC=jnp.array(0.8),
            dD=jnp.array(0.8),
            K_het=jnp.array(20.0),
        )

    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        voltammetry: AbstractVoltammetryTechnique,
        seed: int = 0,
    ):
        key = jr.key(seed)
        generate_key, sampling_key, key = jr.split(key, 3)

        fdm_solver = HeterogeneousReactionFDSolver(voltammetry)

        base_current = fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            0.1,
            key=key,
        )

        def logdensity_fn(params: HeterogenousReactionParams, samples=samples):
            current = fdm_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = HeterogenousReactionParams(
            alpha1=jnp.linspace(0.5, 0.7, NUM_CPUS),
            K1_0=jnp.linspace(5.0, 15.0, NUM_CPUS),
            E1_f=jnp.linspace(0.2, 0.7, NUM_CPUS),
            alpha2=jnp.linspace(0.5, 0.7, NUM_CPUS),
            K2_0=jnp.linspace(5.0, 15.0, NUM_CPUS),
            E2_f=jnp.linspace(0.2, 0.7, NUM_CPUS),
            dB=jnp.linspace(0.8, 1.4, NUM_CPUS),
            dC=jnp.linspace(0.8, 1.4, NUM_CPUS),
            dD=jnp.linspace(0.8, 1.4, NUM_CPUS),
            K_het=jnp.linspace(5.0, 15.0, NUM_CPUS),
        )

        start_time = perf_counter()
        samples, logdensity, info = sampling_algorithm(
            sampling_key, init_params, logdensity_fn
        )

        samples.alpha1.block_until_ready()
        end_time = perf_counter()

        data_file = f"Heterogenous_{sampling_algorithm}_{voltammetry}.npz"

        np.savez_compressed(
            f"./data/{data_file}",
            alpha1=samples.alpha1,
            K1_0=samples.K1_0,
            E1_f=samples.E1_f,
            alpha2=samples.alpha2,
            K2_0=samples.K2_0,
            E2_f=samples.E2_f,
            dB=samples.dB,
            dC=samples.dC,
            dD=samples.dD,
            K_het=samples.K_het,
            logdensity=logdensity,
        )

        print(f"Time Taken: {end_time - start_time:.2f}s")
        print(f"Number of Samples: {len(samples.alpha1.flatten())}")
        print(f"Data Type: {samples.alpha1.dtype}")

        for k, v in info.items():
            print(f"{k}: {v}")
