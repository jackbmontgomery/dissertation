# Dissertation Guide

**Title:** Differentiable Simulation for the Electrochemical Inverse Problem

**Examiner assumptions:**

- A lot: finite difference methods, applied mathematics
- A bit: Bayesian statistics
- Almost nothing: electrochemistry

**Goal:** Develop a differentiable PDE solver for voltammetric simulation and
integrate it with gradient-based Bayesian inference (NUTS) to solve the
electrochemical inverse problem. Three contributions: (1) adjoint-based
differentiable forward solver, (2) full Bayesian treatment via NUTS giving
posterior distributions, (3) extension to chemically complex reactions in higher
parameter dimensions.

**Key result:** Both NUTS and RWMH recover true parameters well. The advantage
of NUTS is clear when we consider the information in the posterior samples, this
is greater at higher dimensions though not linearly since we see the advantage
reduce when there are high correlations in the parameters. Further, we see
better convergence in the GR statistic for NUTS which means we obtain stable
posterior estimates throughout the sampling process.

**Narrative arc:** Each chapter introduces a tool and immediately applies it to
the electron-transfer reaction, so results accumulate throughout. The final two
chapters apply the complete framework to harder reactions concisely, without
repeating exposition.

## Constraints

- `texcount`: We use the "Words in text" field not the "Sum count"

## What I can still add

- Some specific measures for the speed up of the adjoint vs tape

  - Custom VJP

  ```
  In [19]: %timeit ele_fwd(ele_params)
  3.38 ms ± 4.53 μs per loop (mean ± std. dev.
  of 7 runs, 100 loops each)

  In [20]: %timeit ele_grad(ele_params)
  7.25 ms ± 64.3 μs per loop (mean ± std. dev.
  of 7 runs, 100 loops each)

  In [21]: %timeit het_fwd(het_params)
  18 ms ± 37.9 μs per loop (mean ± std. dev. of
   7 runs, 100 loops each)

  In [22]: %timeit het_grad(het_params)
  43.7 ms ± 668 μs per loop (mean ± std. dev. o
  f 7 runs, 10 loops each)
  ```

  - Standard AD

  ```
  In [19]: %timeit ele_fwd(ele_params)
  3.38 ms ± 38.3 μs per loop (mean ± std. dev.
  of 7 runs, 100 loops each)

  In [20]: %timeit ele_grad(ele_params)
  15.8 ms ± 209 μs per loop (mean ± std. dev. o
  f 7 runs, 100 loops each)

  In [21]: %timeit het_fwd(het_params)
  18 ms ± 84.6 μs per loop (mean ± std. dev. of
   7 runs, 100 loops each)

  In [22]: %timeit het_grad(het_params)
  481 ms ± 2.09 ms per loop (mean ± std. dev. o
  f 7 runs, 10 loops each)
  ```

- Need to include that the implementation used a Thomas algorithm written in C++
  for efficiency, but this was only possible because of the custom VJP we have
  derived since we cannot tape over a foreign function.
- Explaining the choice of discritisation more rigorously (but where to put
  this? Because am I going to verify the discritisation before introducing the
  solver)
- Some explanation for the second order current used
- Better explanation for the Rhat being below 1
- Reference the 1.01 threshold in the Rhat figures

---

## Chapter 1: Introduction

**Purpose:** Frame the problem, motivate each methodological layer, state what
was achieved. No numerical results -- each chapter presents its own.

**Arc:** Inverse problem (recover parameters from noisy current) -> forward
problem well understood, inverse is ill-posed (correlations, small influences
depending on scan rate/kinetic regime) -> point estimates insufficient, need
full posterior -> MCMC expensive, gradients via adjoint accelerate both
initialisation and sampling -> demonstrated on electron transfer (3 params),
heterogeneous (7), adsorption (9).

**Positioning:**

- Kitchin et al. (2025) -- "fifth paradigm." This work is a concrete instance.
- Chen et al. (2026) -- we complement: Bayesian not just optimisation; custom
  adjoint not taping.
- Gavaghan et al. (2018) -- we build on this with gradient-based methods.
- Compton & Banks -- _Understanding Voltammetry_.

**Tone:** Qualitative -- no numerical results. Finalise once all reactions
complete.

---

## Chapter 2: Electrochemical Systems

**Job:** Teach the markers the electrochemistry. Only chapter that does this.

**Key points:**

- Physical picture of CV: kinetic vs diffusional control -> peak shape
- Butler-Volmer as the constitutive law (standard first treatment; note other
  models exist e.g. MHC -- pick up in conclusions as further work)
- Dimensionless BVP as the chapter's deliverable -> normalisation table is the
  contract with Ch 3
- Equal-diffusion reduction and K_red/K_ox shorthand

**Core references:** Compton & Banks, Dickinson & Wain (2020), Crank (1979)

---

## Chapter 3: Numerical Methods

**Job:** Discretise the forward problem, verify it, then show why inference is
hard.

**Key points:**

- Exponentially expanding mesh (non-standard, explain properly)
- BV entering the first row (the non-standard, parameter-dependent part)
- Standard FD material stated concisely -- the markers know this
- Verify against analytical solutions: Randles-Sevcik peak current (reversible
  limit), Compton & Banks (irreversible limit)
- Define forward map P: parameters -> predicted current
- Parameter effects and kinetic regimes -> motivation for Bayesian approach
- Scan rate and its choice for the optimal chemical: The scan rate is chosen for
  each reaction mechanism to ensure that the kinetic processes of interest
  operate in a regime where the current response is sensitive to the target
  parameters.

**Arc:** Build solver -> verify -> define P -> show parameters are
correlated/degenerate -> Ch 4

**Core references:** Compton & Banks, Sevcik (1948), Randles (1948)

---

## Chapter 4: Bayesian Approach

**Job:** Frame the inverse problem probabilistically, run RWMH, show it works
but wastes computation on burn-in.

**Key points:**

- Noise model: observed current = P(phi) + eps, sigma fixed as percentage of
  peak current (Morris 2013). Not a tuned parameter -- a design choice for
  synthetic data generation.
- Bayes -> uniform prior -> state l(phi) directly (don't derive intermediate
  steps)
- RWMH: explain the algorithm, tuning Sigma, 23% acceptance heuristic
- **Results:** marginal posteriors recover true values. Corner plot (3x3) --
  introduce this visualisation as baseline. Burn-in scatter plot showing
  computational waste. We have vanilla failing when the scan rate is too low.
- Key observation: each likelihood = one forward solve. Burn-in is expensive.
  Gradients could locate the mode and improve proposals -> Ch 5

**Arc:** Define inference problem -> RWMH -> works but slow -> gradients help ->
Ch 5

**Core references:** Gavaghan et al. (2018), Hastings (1970), Gelman Roberts &
Gilks (1997), Morris (2013)

---

## Chapter 5: Differentiating Through the Forward Solver

**Job:** Core technical contribution. Efficient grad(l) via adjoint of the
tridiagonal solve.

**Key points:**

- Discretise-then-optimise: exact gradient of the discrete computation
- Reverse-mode AD natural (scalar output), but naive taping is O(mn) memory
- Adjoint: A^T lambda = xbar is itself tridiagonal -> O(n) per timestep, cache A
  and x only. Abar = -lambda x^T for matrix entry gradients. Implemented via
  Custom VJP rule in JAX
- Generalises to pentadiagonal -- state briefly, payoff in Ch 7
- **Optimisation results:** ADAM vs CMA-ES on equalised forward-solve budget, 32
  random starts, two noise levels. ADAM converges faster, lower variance.

**Arc:** Need gradients -> naive AD too expensive -> tridiagonal adjoint ->
demonstrate via optimisation -> Ch 6

**Core references:** Griewank & Walther (2008), Kidger (2021), Kingma & Ba
(2014), Hansen & Ostermeier (2001), JAX (Bradbury et al.)

---

## Chapter 6: Hamiltonian Monte Carlo

**Job:** Introduce HMC/NUTS, describe the full inference workflow, show it
outperforms RWMH.

**Key points:**

- Physical intuition for HMC -- new to markers, spend words here
- H(q,p) = U(q) + K(p), leapfrog -- concise, markers know ODEs
- Each leapfrog step = one forward + adjoint solve
- NUTS: adaptive trajectory length, no hand-tuned L
- **Canonical transformation:** alpha -> logit, K0 -> log, theta_f
  unconstrained. Window adaptation tunes mass matrix and step size.
- **Workflow:** (1) LHS -> diverse starts, (2) optimise with fixed budget ->
  high density, (3) best chain -> window adapt -> mass matrix (NUTS) and
  proposal covariance (RWMH, 23%), (4) sample NUTS and RWMH for equal wall-time
- **Results:** posteriors overlaid with RWMH. Simulated current from posterior
  means. ESS -- NUTS wins.

**Note:** Neglidible influence of alpha in reversible regime (from Ch 3) -- show
how the posterior handles this if space permits, otherwise note as further work.

**Arc:** HMC theory -> transformation -> workflow -> results -> NUTS wins ->
apply to harder reactions

**Core references:** Neal (2011), Betancourt (2018), Hoffman & Gelman (2014),
Gelman Roberts & Gilks (1997)

---

## Chapter 7: Heterogeneous Reaction

**Job:** First application to a harder system. Concise -- no re-exposition.

**What's new:**

- Reaction: A + e- <=> B, B ->(k_het) C, C + e- <=> D
- Four coupled diffusion equations
- Pentadiagonal reordering trick: index A,B from N->1, C,D from 1->N so K_het
  coupling is adjacent -> pentadiagonal not nonadiagonal. Include band-structure
  schematic.
- Pentadiagonal adjoint generalises trivially from Ch 5
- 7 parameters: alpha_1, K0_1, theta_f_1, alpha_2, K0_2, theta_f_2, K_het
- **AC voltammetry:** same workflow with sinusoidal perturbation. Show
  posteriors tighten vs DC. Gavaghan et al. (2018) result. And mention briefly
  the non-dimensionalisation scheme
- Frame as simplification of Ch 8 (irreversible surface step, no explicit
  surface concentrations)

**Results:** Analytical verification. Selected pairwise corner plot. ESS (gap
widens 3->7). Current fit. AC vs DC posterior comparison.

**Core references:** Compton & Banks, Gavaghan et al. (2018)

---

## Chapter 8: Adsorption Reaction

**Job:** Most complex system. Novelty is nonlinear coupling and AD compatibility
argument.

**What's new:**

- Langmuir adsorption/desorption with surface coverage ODE coupled to diffusion
  PDE at boundary
- Nonlinear terms prevent direct banded solve
- Three solvers: Newton (accurate but AD-incompatible -- while loop), explicit
  linearisation (AD-compatible), Britz backward implicit
- **Key choice:** ground truth from Newton, inference with explicit scheme.
  Justify by showing equivalent current traces.
- 9 parameters -- highest dimensional case
- Frame as rigorous version of Ch 7

**Results:** Linearisation comparison. 9-param posteriors (selected marginals).
ESS (largest gap -- dimension story completes). Current fit.

**Core references:** Britz -- _Digital Simulation in Electrochemistry_, Compton
& Banks

---

## Chapter 9: Conclusions

**Job:** Brief summary, substantive further work, connect back to introduction.

**Summary (~1 paragraph):** Three contributions delivered. NUTS advantage over
RWMH grows with dimensionality.

**Further work:**

- _Experimental data:_ All results synthetic. Real voltammograms bring model
  misspecification (capacitive current, ohmic drop, electrode roughness).
- _Alternative kinetic models:_ BV -> MHC extension. Bayesian model comparison
  BV vs MHC. Connects to Ch 2 note and Chen et al.
- _Neural operator emulator:_ Solver computes full C(X,T) but inference uses
  only surface flux J(T). Train neural operator on forward solver.
  Differentiable by construction, orders of magnitude faster, NUTS-compatible.
- _Alternative samplers:_ MCLMC, other adaptive HMC variants.

**Closing (~1-2 sentences):** Callback to Kitchin "fifth paradigm." Framework is
not specific to the reactions studied.

**Core references:** Kitchin et al. (2025), Chen et al. (2026)

```

```
