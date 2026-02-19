import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.adaptation.laps import laps
from jaxtyping import PRNGKeyArray

from src.fdm import EMechanismFDMSolver
from src.params import EMechanismFDMParams
from src.utils import generate_noisy_samples
from src.voltammetry import LinearSweepDC


def main():
    key = jr.key(0)
    generate_key, sampling_key, key = jr.split(key, 3)

    voltammetry = LinearSweepDC()
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
        # jax.debug.print("{x}", x=params)
        current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    def sample_init(key: PRNGKeyArray) -> EMechanismFDMParams:
        key_alpha, key_K0, key_E0, key_dB = jr.split(key, 4)
        alpha = jr.uniform(key_alpha, minval=0.3, maxval=0.7)
        K0 = jr.uniform(key_alpha, minval=5.0, maxval=50.0)
        E0 = jr.uniform(key_alpha, minval=0.0, maxval=3.0)
        dB = jr.uniform(key_alpha, minval=0.5, maxval=1.5)
        return EMechanismFDMParams(alpha=alpha, K0=K0, E0=E0, dB=dB)

    num_chains = 2000
    num_steps1, num_steps2 = 10, 10

    mesh = jax.sharding.Mesh(jax.devices()[:1], "chains")

    print("Number of devices: ", len(jax.devices()))

    info, grads_per_step, _acc_prob, samples = laps(
        logdensity_fn=logdensity_fn,
        sample_init=sample_init,
        ndims=4,
        num_steps1=num_steps1,
        num_steps2=num_steps2,
        num_chains=num_chains,
        mesh=mesh,
        rng_key=jax.random.key(0),
        early_stop=False,
        diagonal_preconditioning=True,
        steps_per_sample=10,
        r_end=0.01,
        diagnostics=False,
        superchain_size=1,
    )

    print(samples.position.alpha.shape)


main()
