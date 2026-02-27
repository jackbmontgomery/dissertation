from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import AbstractVoltammetryTechnique

from .base import AbstractSamplingExperiment


class ElectronReactionSamplingExperiment(AbstractSamplingExperiment):
    @property
    def true_parameters(self) -> ElectronReactionParams:
        return ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(5.0),
            Ef=jnp.array(0.5),
            dB=jnp.array(0.8),
        )

    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        voltammetry: AbstractVoltammetryTechnique,
        seed: int = 0,
    ):
        key = jr.key(seed)
        generate_key, sampling_key, key = jr.split(key, 3)

        fdm_solver = ElectronReactionFDSolver(voltammetry)

        base_current = fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            0.1,
            key=key,
        )

        def logdensity_fn(params: ElectronReactionParams, samples=samples):
            current = fdm_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = ElectronReactionParams(
            alpha=jnp.linspace(0.5, 0.7, NUM_CPUS),
            K0=jnp.linspace(2.0, 8.0, NUM_CPUS),
            Ef=jnp.linspace(0.2, 0.8, NUM_CPUS),
            dB=jnp.linspace(0.6, 1.0, NUM_CPUS),
        )

        start_time = perf_counter()
        samples, logdensity, info = sampling_algorithm(
            sampling_key, init_params, logdensity_fn
        )

        data_file = f"E_{sampling_algorithm}_{voltammetry}.npz"
        np.savez_compressed(
            f"./data/{data_file}",
            alpha=samples.alpha,
            K0=samples.K0,
            E0=samples.E0,
            dB=samples.dB,
            logdensity=logdensity,
        )

        end_time = perf_counter()
        print(f"Time Taken: {end_time - start_time:.2f}s")
        print(f"Number of Samples: {len(samples.alpha.flatten())}")
        print(f"Data Type: {samples.alpha.dtype}")

        for k, v in info.items():
            print(f"{k}: {v}")
