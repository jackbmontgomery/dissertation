import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax.numpy as jnp
import jax.random as jr
from jax import pmap, vmap
from jax.lax import scan
from jaxtyping import PRNGKeyArray


def generate_samples(key: PRNGKeyArray):
    return jr.normal(key, shape=(20,))


vmap_generate_samples = vmap(generate_samples)

key = jr.key(42)
keys = jr.split(key, (8, 10))
samples = pmap(vmap_generate_samples, in_axes=0)(keys)
print(samples.shape)
