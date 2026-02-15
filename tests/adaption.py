import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


from time import perf_counter

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.diagnostics import potential_scale_reduction
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.sampling import (
    NutsSamplingAlgorithm,
)
from src.utils import generate_noisy_samples
from src.voltammetry import LinearSweepDC


def main():
    voltammetry = LinearSweepDC()

    # --- NUTS ---
    # sampling_algorithm = NutsSamplingAlgorithm(
    #     100, 1e-2, inv_mass_matrix=jnp.array([0.028, 0.028, 0.028, 0.028])
    # )

    # --- Sampling ---
    key = jr.key(42)
    generate_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = EMechanismFDMSolver(voltammetry)

    true_params = EMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(0.5),
    )

    base_current = fdm_solver.solve(true_params)

    samples = generate_noisy_samples(
        10,
        base_current,
        0.1,
        key=key,
    )

    def log_density(params: EMechanismFDMParams, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = EMechanismFDMParams(
        alpha=jnp.array(0.3),
        K0=jnp.array(20.0),
        E0=jnp.array(0.0),
        dB=jnp.array(0.8),
    )

    approx_key, sample_key = jr.split(key)

    print("--- Approximating ---")

    adapt = blackjax.pathfinder_adaptation(
        blackjax.nuts, log_density, initial_step_size=1e-2
    )

    (state, parameters), info = adapt.run(sample_key, init_params, num_steps=20)

    print(parameters)

    sampling_algorithm = NutsSamplingAlgorithm(100, **parameters)

    samples, info = sampling_algorithm(sampling_key, state.position, log_density)

    samples = jnp.squeeze(jnp.array([jax.tree.leaves(samples)]), 0)
    x = jax.vmap(potential_scale_reduction)(samples)
    print(x)


if __name__ == "__main__":
    main()
