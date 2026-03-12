import jax.numpy as jnp
import jax.random as jr
import numpy as np
from chex import PRNGKey

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicAC, CyclicDC, VoltammetryType

from .base import AbstractSamplingExperiment


def create_init_params(key: PRNGKey, num_chains: int):
    k1, k2, k3 = jr.split(key, 3)

    alpha_vals = jnp.linspace(0.5, 0.7, num_chains)
    K0_vals = jnp.linspace(5.0, 20.0, num_chains)
    Ef_vals = jnp.linspace(0.0, 1.0, num_chains)

    alpha = jr.permutation(k1, alpha_vals)
    K0 = jr.permutation(k2, K0_vals)
    Ef = jr.permutation(k3, Ef_vals)

    return ElectronReactionParams(
        alpha=alpha,
        K0=K0,
        Ef=Ef,
    )


class ElectronSamplingExperiment(AbstractSamplingExperiment):
    @property
    def true_parameters(self) -> ElectronReactionParams:
        return ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(10.0),
            Ef=jnp.array(0.5),
        )

    def run(
        self,
        sampling_algorithm: AbstractSamplingAlgorithm,
        noise: float,
        sigma: int,
        voltammetry_type: VoltammetryType,
        seed: int = 0,
    ):
        key = jr.key(seed)

        if voltammetry_type == "DC":
            voltammetry = CyclicDC(sigma=sigma)
        else:
            voltammetry = CyclicAC(sigma=sigma)

        param_key, sampling_key, key = jr.split(key, 3)

        fdm_solver = ElectronReactionFDSolver(voltammetry)

        _, base_current = fdm_solver.solve(self.true_parameters)

        samples = generate_noisy_samples(
            10,
            base_current,
            noise,
            key=key,
        )

        def logdensity_fn(params: ElectronReactionParams, samples=samples):
            _, current = fdm_solver.solve(params)
            return -jnp.sum((samples - current) ** 2)

        init_params = create_init_params(param_key, NUM_CPUS)
        samples, logdensity = sampling_algorithm(
            sampling_key, init_params, logdensity_fn
        )
        data_file = (
            f"E_{voltammetry_type}_{sampling_algorithm}_{noise:.2f}_{sigma:.0f}.npz"
        )
        np.savez_compressed(
            f"./data/{data_file}",
            alpha=samples.alpha,
            K0=samples.K0,
            Ef=samples.Ef,
            logdensity=logdensity,
        )
