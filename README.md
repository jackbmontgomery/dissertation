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
- Investigate some convergence measures in the sampling experiment (ESS and Gelman and Rubin)
- Investigate the performance with varying degrees of noise 
- Correlation between the difference parameters
- Plot of non-linear approximation error: I can compute what the non-linear value is using the Newton solver. I can then store the value of the non-linear approximation for the two linear approximations. I can then compare the "ground truth" of Newton to the linear approximations to see where the explicit method diverges for large Kplus


## Questions
1. When should I start counting the samples I am obtaining from the MCs? I need a diagnostic for convergence of the n-step Markov transition to the true distribution. (Vats, 2020). In other words, a diagnostic for the burn-in period.
2. Then I also need a diagnostic for the convergence of the sample statistics.
- Together, these will give me a more principled way of determining when I have collected actual samples not just playing around with the look of the distributions and being happy with that.
3. A principled way of determining the hyper parameters of the difference sampling algorithms. For the MCMC, I just need the sigma matrix which can be chosen to give myself a certain acceptance rate (~0.23). For the NUTS, I can use the VI to estimate the step size and the inverse mass matrix.

## For using pmap on single CPU
```python
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```
