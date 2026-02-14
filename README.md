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
- Cyclic voltammetry seems to make this almost a trivial task to get the posterior distribution. The predictions are very accurate and with very low variance. But not necessarily interesting, can make the assumption that we are only using linear sweep techniques.
- The step size sampling requires the plausible parameters to be on a similar scale
- I need to investigate some convergence measures in the sampling experiment

## For using pmap on single CPU
```python
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```
