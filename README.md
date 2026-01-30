# Dissertation

# For using pmap on single CPU

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

Errors:
- TypeError: Argument '<function log_density at 0x125b4b6a0>' of type <class 'function'> is not a valid JAX type.
