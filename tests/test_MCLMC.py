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

    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    init_state = blackjax.mcmc.mclmc.init(
        position=init_params, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    init_state, adaption_params, _ = blackjax.mclmc_find_L_and_step_size(
        kernel, 100, init_state, adjust_key, diagonal_preconditioning=False
    )

    mclmc = blackjax.mclmc(
        logdensity_fn,
        L=adaption_params.L,
        step_size=adaption_params.step_size,
    )

    mclmc_kernel = jit(mclmc.step)

    start_time = perf_counter()
    state, infos = inference_loop(sampling_key, mclmc_kernel, init_state, 1000)
    end_time = perf_counter()
    print(f"Time Taken: {end_time - start_time:.4f}")
    print(infos)
    # print("Is Accepted:", jnp.mean(infos.is_accepted))
    # print("Average Integration Steps", jnp.mean(infos.num_integration_steps))
    plt.hist(state.position.alpha.flatten())
    plt.show()


main()
