# Presentation Notes

## Slide 3: Redox reaction and the electrochemical cell
- Reduction of A to B
- Oxidation of B to A
- Potentiostat to apply potential to the working electrode, counter electrode to complete the electric circuit
- Potentiostat to measure the current in the circuit

## Slide 4: Voltammagram
- Triangular wave form of cyclic voltammetry
- Duck-like cathodic and anodic peaks
- Assuming that the size of the electrode is large compared with the distance at which the concentration can diffusion on the timescale of the experiment: Macroelectrode assumption
- Single dimension of the diffusion acting perpendicular to the electrode

## Slide 5: Mathematical Description
- Fick's Second law to describe how concentration will change due to the diffusion
- Butler-Volmer formalism to relate the applied potential to the change in concentration
- Current as the experimental artifact

## Slide 6: Dimensionless Mathematical Description
- Only line to really note is that the current is just a scaling of the dimensionless flux. So we will record the flux in the numerical simulations and we can scale that to relevant experimental setups. 

## Slide 7: Parameters
- The rate constant with each component without the concentrations is the rate of reductions and oxidation
- alpha is the **charge transfer coefficient** which specifies whether the process favours the reduction or oxidations.
- Dimensionless rate constant which characterises the reversibility of the process
- Potential at which the rate of oxidation and reduction is equal

## Slide 8: Parameter Effects
- alpha: higher alpha is greater means cathodic peak and the lower the anodic peak
- K0: greater means more symmetry
- Ef: some translation

## Slide 9: Forward and Inverse problem
- In general, the finite difference solver maps to the solution of the PDE but in this context we only care about the flux wave form
- Generally, there does not exist a single parameter set that will minimise this. This warrants the use of probabilistic methods. Specifically, we will use Bayesian methods

## Slide 10: Bayesian Inference
- P is equal to the likelihood in this case
- Description of drawing the uniform rv between 0 and 1

## Slide 11: Markov Chain Monte Carlo

## Slide 12: Random-Walk Metropolis Hastings
- Proposal is a random step away from the current state
- Sigma is the covariance matrix that will control how far we deviate from the current state. Generally tuned to get an acceptance of 0.23
- Notice the tails on the sample distributions

## Slide 9: Burn-in
- See the algorithm moving into areas of high density
- Metropolis adjusted algorithms do not converge very quickly
- Motivates the use of other algorithm to move the chain into areas of high density and then start sampling
- We know that these values are possible but they are very unlikely. In this kind of Bayesian setup, a sample is not very cheap. We need to run a whole forward solve in order to obtain the likelihood.

## Slide 10: Differentiable Simulator
- Go back to the description of the finite difference method

## Slide 11: Path to high likelihood
- Comparison between steps of gradient descent and samples from the MCMC
- Similarly we can compare this to gradient free methods where we find similar performance enhancements
- Gradient-free optimisers also exist and we use them for comparison in the dissertation but in general they are not as robust to initial conditions or as efficient

## Slide 12: Hamiltonian Monte Carlo
- Description of this method will go beyond the time of the presentation 
- ESS: how quickly do our autocorrelations fade

## Slide 13: ESS
- How much information we have in our samples

## Slide 14: Further Results
- General note about computational resourses
