from time import perf_counter

import blackjax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit

from src import sampling
from src.fdm import EMechanismFDMSolver
from src.params import ECirreMechanismFDMParams, EMechanismFDMParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm, inference_loop
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC


def main():
    key = jr.key(0)
    generate_key, sampling_key, key = jr.split(key, 3)

    voltammetry = CyclicDC()

    fdm_solver = EMechanismFDMSolver(voltammetry)

    true_params = EMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(1.2),
    )

    base_current = fdm_solver.solve(true_params)

    samples = generate_noisy_samples(
        10,
        base_current,
        0.1,
        key=key,
    )

    def logdensity_fn(params: EMechanismFDMParams, samples=samples):
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = EMechanismFDMParams(
        alpha=jnp.array(0.5),
        K0=jnp.array(10.0),
        E0=jnp.array(1.8),
        dB=jnp.array(1.1),
    )

    init_key, adjust_key, sampling_key = jr.split(sampling_key, 3)

    sigma = jnp.array([0.005, 0.005, 0.005, 0.005])
    rmh = blackjax.rmh(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))

    rmh_kernel = jit(rmh.step)
    init_state = rmh.init(init_params, init_key)

    start_time = perf_counter()
    state, infos = inference_loop(sampling_key, rmh_kernel, init_state, 1000)
    state.position.alpha.block_until_ready()
    end_time = perf_counter()
    print(f"Time Taken: {end_time - start_time:.4f}")
    print(jnp.average(infos.acceptance_rate))
    # print("Is Accepted:", jnp.mean(infos.is_accepted))
    # print("Average Integration Steps", jnp.mean(infos.num_integration_steps))
    plt.hist(state.position.alpha.flatten())
    plt.show()


main()
