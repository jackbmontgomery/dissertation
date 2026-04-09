# Dissertation: Differentiable Simulation for Electrochemical Inverse Problems

The examiners will know:

1. **A lot** about finite difference methods and applied mathematics
2. **A bit** about Bayesian statistics
3. **Almost nothing** about electrochemistry

## Overview

The dissertation is organised around the cyclic voltammetry inverse problem for
a single-electron redox reaction, which serves as the setting in which every
methodological component is introduced and motivated. We begin by formulating
the forward problem (diffusion PDE, Butler–Volmer kinetics,
non-dimensionalisation, numerical discretisation), then frame the inverse
problem as Bayesian inference with MCMC sampling, showing the computational cost
of burn-in and the limitations of gradient-free proposals. This motivates the
development of adjoint-based differentiation through the forward solver, which
in turn enables gradient-based initialisation and Hamiltonian Monte Carlo
sampling. Chapters 2 to 6 are introductory and substantive: they develop a key
tool and immediately applies it to the electron-transfer problem, so results
accumulate throughout rather than being deferred to a final chapter. With the
full methodology established, the heterogeneous and adsorption reactions are
treated more concisely, demonstrating that the framework extends to
higher-dimensional, chemically richer systems without repeating the expository
groundwork.

## Chapter 1: Introduction

STATUS: Draft and missing heterogenous and adsorption results

MESSAGE: We develop a differentiable PDE solver for voltammetric simulation and
integrate it with gradient-based Bayesian inference to solve the electrochemical
inverse problem. The approach yields three contributions: (1) a fast,
adjoint-based differentiable forward solver for electrochemical PDEs, (2) a full
Bayesian treatment via HMC that provides posterior distributions rather than
point estimates, and (3) demonstration that the framework extends to chemically
complex reactions (heterogeneous, adsorption) in higher parameter dimensions.

## Chapter 2: Electrochemical Systems

Figures:

1. Electrochemical Cell
2. Applied potenetial and voltammagram
3. Schematic figure of the 1D reaction

Still to add:

1. Note about the different kinetic regimes

LINK TO NEXT: No nice link to next chapter

## Chapter 3: Numerical Methods

- Analytical results comparison

LINK TO NEXT: Effect of the parameters and the difficulty in the reversible
reigime sets up a probabilisitc approach

## Chapter 4: Bayesian Approach

- Increase noise and look at the variance of the posterior

LINK TO NEXT: Tail artifacts and the burn in period sets up the use of more
efficient methods to find high density.

## Chapter 5: Differentiating

- Change the figure and caption to be about the comparison with RWMH with
  optimisation techniques. How CMA-ES and ADAM perform well but as we will see.
  When we move to higher dimensions and more complicated problems then ADAM far
  out performs CMA-ES.

LINK TO NEXT: No nice link

## Chapter 6: Hamiltonian Monte Carlo

Still to add:

- Sampling results
