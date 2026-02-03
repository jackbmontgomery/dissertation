import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

from time import perf_counter
from typing import Literal

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
from jax import vmap
from jax.scipy.special import logit

from src.experiment import LinearSweepACMacroBand, LinearSweepDCMacroBand
from src.fdm_discretisation import ButlerVolmerFDMDiscretisation1D, uniform_discretise
from src.pde_parameters import ButlerVolmerInverseParameters, bv_inverse_to_physical
from src.sampling import generate_noisy_samples, mclmc_sampling, rw_sampling
from src.simulate import create_fdm_current_simulator


def main(
    experiment_type: Literal["dc", "ac"],
    sampling_algorithm: Literal["rw", "mclmc"],
    n_samples: int = 20_000,
    seed: int = 42,
):
    print(f"Sampling: {sampling_algorithm}, Experiment: {experiment_type}")
    key = jr.key(seed)

    true_params = ButlerVolmerInverseParameters(
        a=logit(0.6), k0=jnp.log(100.0), e0=5.0 * jnp.arctanh(2.0 / 10.0)
    )

    phy_params = bv_inverse_to_physical(true_params)

    if experiment_type == "dc":
        experiment = LinearSweepDCMacroBand()
    elif experiment_type == "ac":
        experiment = LinearSweepACMacroBand()
    else:
        raise Exception("Invalid Experiment Choice")

    T, X = uniform_discretise(experiment)
    print(f"T:{T.shape},X:{X.shape}")

    potentials = vmap(experiment.potential)(T)

    fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, T)
    c_init = jnp.ones_like(X)

    simulate_current = create_fdm_current_simulator(
        c_init, potentials, fdm_discretisation, X
    )

    samples = generate_noisy_samples(10, simulate_current, phy_params, 0.1, key=key)

    def log_density(params: ButlerVolmerInverseParameters, samples=samples):
        phy_params = bv_inverse_to_physical(params)
        pred = simulate_current(phy_params)
        return -jnp.sum((samples - pred) ** 2)

    bfgs = optx.BFGS(rtol=1e-4, atol=1e-4)
    init_params = ButlerVolmerInverseParameters(
        a=logit(0.3), k0=jnp.log(80.0), e0=jnp.arctanh(0.0)
    )
    solution = optx.minimise(lambda params, _: -log_density(params), bfgs, init_params)
    sampling_params = solution.value

    x = bv_inverse_to_physical(solution.value)
    print("Params", "alpha", x.alpha, "kappa", x.kappa0, "eps", x.eps0)

    if sampling_algorithm == "rw":
        sample_key, key = jr.split(key, 2)

        sampling_fn = rw_sampling

    elif sampling_algorithm == "mclmc":
        num_cpus = multiprocessing.cpu_count()

        sample_key = jr.split(key, num_cpus)

        sampling_params = ButlerVolmerInverseParameters(
            a=jnp.full((num_cpus,), sampling_params.a),
            k0=jnp.full((num_cpus,), sampling_params.k0),
            e0=jnp.full((num_cpus,), sampling_params.e0),
        )

        n_samples = n_samples // num_cpus

        sampling_fn = mclmc_sampling
    else:
        raise Exception("Invalid Sampling Algorithm")

    print("Sampling Started")
    start_time = perf_counter()
    states, infos = sampling_fn(sample_key, n_samples, sampling_params, log_density)
    _ = states.position.a.block_until_ready()
    end_time = perf_counter()
    print("Sampling Done. Time taken:", end_time - start_time)
    avg_acceptance = jnp.mean(infos.is_accepted)
    print("Avg Acceptance:", avg_acceptance)

    alpha = jnn.sigmoid(states.position.a.flatten())
    kappa0 = jnp.exp(states.position.k0.flatten())
    eps0 = 10.0 * jnp.tanh(states.position.e0.flatten() / 5.0)

    np.savez_compressed(
        f"./data/{sampling_algorithm}_{experiment_type}.npz",
        alpha=alpha,
        kappa0=kappa0,
        eps0=eps0,
    )


if __name__ == "__main__":
    # main(experiment_type="ac", sampling_algorithm="rw", n_samples=40_000)
    main(experiment_type="ac", sampling_algorithm="mclmc", n_samples=8_000)
    # main(experiment_type="dc", sampling_algorithm="rw", n_samples=40_000)
    main(experiment_type="dc", sampling_algorithm="mclmc", n_samples=8_000)
