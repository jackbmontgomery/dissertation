import multiprocessing
import os
from time import perf_counter

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optimistix as optx
from jax import vmap
from jaxtyping import PRNGKeyArray

from src.fdm_dep import AbstractFDMSolver, MacroElectrodeFDMSolver
from src.params import MacroElectrodeParams
from src.sampling import mclmc_sampling, rw_sampling
from src.voltammetry import LinearSweepDC


def generate_noisy_samples(
    num_samples: int,
    params: MacroElectrodeParams,
    sigma: float,
    fdm_solver: AbstractFDMSolver,
    *,
    key: PRNGKeyArray,
):
    _solution, current = fdm_solver.solve(params)

    def add_noise(r_key: PRNGKeyArray, current=current):
        noisy_current = current + jr.normal(r_key, shape=current.shape) * sigma
        return noisy_current

    keys = jr.split(key, num_samples)
    samples = vmap(add_noise)(keys)
    return samples


def main(
    seed: int = 42,
    experimental_samples: int = 10,
    experimental_noise: float = 0.1,
    n_samples: int = 4000,
):
    key = jr.key(seed)
    generate_key, sampling_key, key = jr.split(key, 3)

    voltammetry = LinearSweepDC()

    fdm_solver = MacroElectrodeFDMSolver(voltammetry, 1e-2, 5e-2)

    experimental_params = MacroElectrodeParams(
        alpha=jnp.array(0.6), kappa=jnp.array(100.0), epsilon=jnp.array(2.0)
    )

    samples = generate_noisy_samples(
        experimental_samples,
        experimental_params,
        experimental_noise,
        fdm_solver,
        key=key,
    )

    def log_density(params: MacroElectrodeParams, samples=samples):
        _solution, current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
    init_params = MacroElectrodeParams(
        alpha=jnp.array(0.3), kappa=jnp.array(80.0), epsilon=jnp.array(0.0)
    )
    solution = optx.minimise(lambda params, _: -log_density(params), bfgs, init_params)
    sampling_init_params = solution.value

    print("Sampling Started")
    start_time = perf_counter()
    states, infos = mclmc_sampling(
        sampling_key, n_samples, sampling_init_params, log_density
    )
    _ = states.position.a.block_until_ready()
    end_time = perf_counter()
    print("Sampling Done. Time taken:", end_time - start_time)
    avg_acceptance = jnp.mean(infos.is_accepted)
    print("Avg Acceptance:", avg_acceptance)


if __name__ == "__main__":
    main()
