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
- Must be careful of only training on a mesh that is solvable fast enough because this might not be accurate. Especially for the idea that I have a fine mesh to generate a accurate data and then use a more coarse one for sampling
- I need to investigate the reduction of the pentadiagonal sysmtem into the tridigonal system because the `solve` method just takes too long


## For using pmap on single CPU
```python
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```
