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
import jax.tree_util as jtu
import optax
from jax import jit, lax, vmap

from src.fdm import ElectronReactionFDSolver
from src.optimisers import make_adam_optimise
from src.params import ElectronReactionParams
from src.reaction import ElectronReaction
from src.sampling import inference_loop_multiple_chains
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

seed = 0
num_chains = 8

key = jr.key(seed)
key_init, key_warmup, key_experiment, key_optimise, key_inference = jr.split(key, 5)

reaction = ElectronReaction()
voltammetry = CyclicDC()
fd_solver = ElectronReactionFDSolver(voltammetry)

_, base_current = fd_solver.solve(reaction.true_parameters)

experimental_samples = generate_noisy_samples(
    10,
    base_current,
    0.02,
    key=key_experiment,
)


def logdensity_fn(params: ElectronReactionParams):
    _, current = fd_solver.solve(params)
    return -jnp.sum((experimental_samples - current) ** 2)


adam_minimise = make_adam_optimise(
    num_steps=100, log_density=logdensity_fn, learning_rate=1e-1
)

init_params = reaction.create_init_params(key_init, 8)

warmed_up_params, log_densities, _ = vmap(adam_minimise)(init_params)

chain_idx = 0

adaption_params = jax.tree.map(lambda x: x[chain_idx], warmed_up_params)

warmup = blackjax.window_adaptation(
    blackjax.nuts,
    logdensity_fn,
    initial_step_size=1e-2,
    target_acceptance_rate=0.8,
)


(last_state, nuts_params), warmup_info = warmup.run(
    key_warmup,
    adaption_params,
    num_steps=500,
)

# ---- Use adapted params for NUTS sampling ----
key_samples = jr.split(key_inference, num_chains)

nuts = blackjax.nuts(logdensity_fn, **nuts_params)
nuts_kernel = jit(nuts.step)

init_states = vmap(nuts.init)(warmed_up_params)

n_samples = 8_000
samples_per_chain = n_samples // num_chains

start = perf_counter()
states, infos = inference_loop_multiple_chains(
    key_samples, nuts_kernel, init_states, samples_per_chain
)

states.position.alpha.block_until_ready()
print(f"Time Taken: {perf_counter() - start:.3f}")

print(
    "Average Integration Steps: ", jnp.round(jnp.mean(infos.num_integration_steps), 2)
)

print("----- ESS -----")
print(f"alpha: {blackjax.diagnostics.effective_sample_size(states.position.alpha)}")
print(f"K0: {blackjax.diagnostics.effective_sample_size(states.position.K0)}")
print(f"thetaf: {blackjax.diagnostics.effective_sample_size(states.position.thetaf)}")

print("----- GR -----")

print(f"alpha: {blackjax.diagnostics.potential_scale_reduction(states.position.alpha)}")
print(f"K0: {blackjax.diagnostics.potential_scale_reduction(states.position.K0)}")
print(
    f"thetaf: {blackjax.diagnostics.potential_scale_reduction(states.position.thetaf)}"
)
