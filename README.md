# Dissertation

## Chapters
1. Electrochemical Systems
    - Basic terms, Laws (Fick's)
    - Voltammetry
    - Finite Difference
2. Sampling
    1. MCMC
    2. HMC
    3. VI
3. $E$
4. $EC_{irre}$
5. $Second Order$


## Notes
- I am a bit worried about what changes in the Butler-Volmer boundary conditions when we add the other items in solution. Though maybe they don't make a difference because they are not electrochemically active


## For using pmap on single CPU

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


## Creating Pentadiagonal Matrices
```python
def pentadiag(d2, d1, d0, u1, u2):
    return (
        jnp.diag(d0) +
        jnp.diag(d1, k=-1) +
        jnp.diag(d2, k=-2) +
        jnp.diag(u1, k=1) +
        jnp.diag(u2, k=2)
    )

```
