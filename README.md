# Dissertation

- I need to make sure that my inverse function make sense for scaling the space of plausible parameters for all models
- I will do the performance squeezing for the adsorption when I get to that chapter, I need to be in the frame of reference to understand all that code

## Burn-in
When running an MCMC sampler there are convergence guarantees and so the concept of needing to burn in does not really make sense. But in an application when we cannot generate samples very easily -- after all the likelihood evaluation needs a forward pass of an fdm solver. In this case it is clear that just letting the MC fall into the stationary distribution is not really satisfactory. We went to ensure that we are obtaining good, high-density samples from the start. In this case, the concept of burn-in makes far more sense.

### What are we measuring?
- How far away can we start the initial guesses and still get convergence to reasonable likelihoods?
- How fast is this convergence if it happens?

## Sampling Efficiency
Once we have started sampling, how do we know how good our samples are? True we still want to be generating a distribution that will approximate the true posterior but this is actually more of a issue of how well posed the problem is. Experimental design is a large contributor to this well-posed-ness. If we have small scan rate or a small potential sweep we will never have the information to determine the parameters.

No I do not believe that accuracy is a good meausre of the quality of the sampler, this is how much information about the posterior do we obtain during sampling. How correlated are the samples? What is the ESS? And what is the trade-off between increasing these and the computational wall-time?

### What are we measuring? 
- How close is the mean / maximum likelihood estimate to the true parameter?
- What is the effective sample size? How fast are we removing the correlations in our markov chain

## Implementation
1. First stage of this needs to be some comparison of the burn-in. How fast do the gradient-based and gradient-free methods converge. This relates pretty closely to what we see in the Differential Electrochemistry paper.
2. Compare the sampling efficiency of the difference methods, and the accuracy of the samples because ESS does not matter much if it is wrong. This is related to how well-poised the problem is not just the sampling algorithm.
