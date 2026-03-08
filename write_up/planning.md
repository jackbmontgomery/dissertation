# Dissertation Write-up Planning

Title: Differentiable Electrochemical Simulation for Gradient-Based Parameter Inference

Purpose: We can more efficiently and accurately sample posterior distributions for parameters by using gradients of a differentiable simulator 
How: Compare gradinent-free methods proposed in (Gavaghan, 2018) and show that we can make those more computationally efficient.

## Burn-in
- Computational Efficiency: How long does it take?
- Density: How close do we get to the true values
- Robustness: How far away can we start
- Scatter plots
- Log-density over time

## Sampling
- Accuracy: Are we getting a distribution that is close the true value?
- Stability of the moments: Is it luck that we get the estimate we get? Or has it converged?
- ESS: How much information is encoded in the distribution we have sampled?
- Biplots


## Outline

### Chapter 1: Introduction
- Electrochemical reactions
- What we aim to understand by using voltammetry
- Using method of Bayesian inference to approximate distributions over the experimental parameters
- Augment likelihood information with gradients to create methods can more efficiently sample
- Efficiency is measured in term on computational time, Accuracy is measures in terms of the estimated of the parameters and that variance
- Compare gradient based burn-in, gradient based RW MCMC, to gradient-free methods.
- Burn-in comparison: Using CMA-ES or fminsearch (Gavaghan, 2018) against Adam with gradients
- Sampling: ESS and computational time


### Chapter 2: Mathematical Description of Electrochemical Systems
- Experimental Setup of Voltammetry
- Mass Transport and Diffusion: Fick's second law
- Electrode Kinetics: Butler-Volmer Equations
- Non-dimensionalisation

### Chapter 3: Numerical Methods
- Implicit FD Setup
- Expressing this as a mapping between the parameters and the induced current

### Chapter 4: MCMC
- Markov chain and Ergodic theory
- Define the object of the inverve problem in term of the numerical simulator and the likelihood
- Metropolis-Hastings algorithm
- Random-Walk Metroplis Hastings ~ 23% acceptance (Gavaghan, 2018)
- Hamiltonian Monte-Carlo
- Scale of the different parameters

### Chapter 5: Automatic Differentiation
- Comments about the computational graph 
- Reverse-mode and forward-mode (vector jacobian product and the jacobian vector product)
- Define the gradient of the likelihood wrt the parameters of interest


### Chapter 6: Electrode Transfer Experiment
- Why this is important?
- General Intuition for how the parameters work
- Burn-in results: Adam vs fminsearch and CMA-ES
- Sampling-results: HMC vs RW


### Chapter 7: Heterogenuous Reaction
- Why this is important?
- General Intuition for how the parameters work
- Comparison with the same Electrode reaction with how the new parameters effect the current
- Burn-in results: Adam vs fminsearch and CMA-ES
- Sampling-results: HMC vs RW
- Contributions of difference components of the current to try understand what is happening

### Chapter 8: Adsorption Reaction
- Why this is important?
- General Intuition for how the parameters work
- Comparison with the same Electrode reaction with how the new parameters effect the current
- Burn-in results: Adam vs fminsearch and CMA-ES
- Sampling-results: HMC vs RW
- Numerics Comparison: Plot the difference in current and the absolute difference in the non-linear approximation
- Contributions of difference components of the current to try understand what is happening

### Chapter 9: Conclusion
...
