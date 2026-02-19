import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import blackjax
import jax

x = blackjax.adap
num_cores = jax.local_device_count()
print(num_cores)
