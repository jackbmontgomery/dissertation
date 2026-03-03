# Dissertation

## Chapters
1. Electrochemical Systems
    - Basic terms, Laws (Fick's)
    - Voltammetry
    - Finite Difference
2. Sampling
    1. MCMC
    2. HMC
3. $E$

4. $EC_{irre}$
    - See 5.1 in Understanding: Effect of the Delta T * K1
5. $Second Order$


## Notes
- The step size sampling requires the plausible parameters to be on a similar scale
- Investigate some convergence measures in the sampling experiment (ESS and Gelman and Rubin)
- Investigate the performance with varying degrees of noise 
- Correlation between the difference parameters
- Plot of non-linear approximation error: I can compute what the non-linear value is using the Newton solver. I can then store the value of the non-linear approximation for the two linear approximations. I can then compare the "ground truth" of Newton to the linear approximations to see where the explicit method diverges for large Kplus
- Ideally, I would like to just use the Chees burn in with the HMC kernel. I think this would just make the most sesnse with what I am studying.
- But I can compare this to MCHMC because this also has a adaption and I have a feeling this will be a little more efficient to run on my CPU.
- I need to adjust my numerics, the non-linear stuff is not working properly to do my sampling. I need to build up a narritive with the chemical reactions and I think the best way to do this is by going from. Basic Electron Only Reaction -> First Order ECirre -> Hetrogeneous Chemical Process -> Absorbsion

## Questions
1. When should I start counting the samples I am obtaining from the MCs? I need a diagnostic for convergence of the n-step Markov transition to the true distribution. (Vats, 2020). In other words, a diagnostic for the burn-in period.
2. Then I also need a diagnostic for the convergence of the sample statistics.
- Together, these will give me a more principled way of determining when I have collected actual samples not just playing around with the look of the distributions and being happy with that.
3. A principled way of determining the hyper parameters of the difference sampling algorithms. For the MCMC, I just need the sigma matrix which can be chosen to give myself a certain acceptance rate (~0.23). For the NUTS, I can use the VI to estimate the step size and the inverse mass matrix.

## Parameters
- Electron Transfer Reaction

## Ideas:
- I can play with noise levels in the first experiment because they can actually solve it.
- In the second there will be too many parameters to play with
- The final is the ability of the method
- We can plot the contribution from the different components in the flux by using the reduction and oxidation formations of the current
