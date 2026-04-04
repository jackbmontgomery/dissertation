# Dissertation Planning Document

**Title:** Differentiable Simulation for the Electrochemical Inverse Problem

## Examiner Assumptions

Per supervisor guidance, assume the markers know:

1. **A lot** about finite difference methods and applied mathematics
2. **A bit** about Bayesian statistics
3. **Almost nothing** about electrochemistry
4. **Not familiar** with automatic differentiation

This means: electrochemical setup, voltammetry, and the physical
interpretation of Butler–Volmer parameters should be explained
carefully. Finite difference discretisation, tridiagonal solves, and
standard numerical analysis can be condensed. Bayesian concepts (prior,
posterior, likelihood, MCMC) need some explanation but not from first
principles.

---

## Structural Overview

The dissertation is organised around the cyclic voltammetry inverse problem
for a single-electron redox reaction, which serves as the setting
in which every methodological component is introduced and motivated. We
begin by formulating the forward problem (diffusion PDE, Butler–Volmer
kinetics, non-dimensionalisation, numerical discretisation), then frame
the inverse problem as Bayesian inference with MCMC sampling, exposing
the computational cost of burn-in and the limitations of gradient-free
proposals. This motivates the development of adjoint-based
differentiation through the forward solver, which in turn enables
gradient-based initialisation and Hamiltonian Monte Carlo sampling. Each
chapter is simultaneously introductory and substantive: it develops a key
tool and immediately applies it to the electron-transfer problem, so
results accumulate throughout rather than being deferred to a final
chapter. With the full methodology established, the heterogeneous and
adsorption reactions are treated more concisely, demonstrating that the
framework extends to higher-dimensional, chemically richer systems without
repeating the expository groundwork.

---

## Chapter 1: Introduction

**Status:** Draft Done with notes about final conclusions for reactions left out

**Key message:** We develop a differentiable PDE solver for voltammetric
simulation and integrate it with gradient-based Bayesian inference to
solve the electrochemical inverse problem. The approach yields three
contributions of roughly equal weight: (1) a fast, adjoint-based
differentiable forward solver for electrochemical PDEs, (2) a full
Bayesian treatment via HMC that provides posterior distributions rather
than point estimates, and (3) demonstration that the framework extends
to chemically complex reactions (heterogeneous, adsorption) in higher
parameter dimensions.

**Must mention:**

- *Problem setting:* Voltammetry measures current as a function of
  applied potential; the inverse problem is to recover the physical
  parameters (rate constants, transfer coefficients, formal potentials)
  from noisy current data. This is ill-posed: multiple parameter
  configurations can produce similar voltammograms, and some parameters
  become non-identifiable in certain kinetic regimes.

- *Forward problem briefly:* The current response is governed by a
  diffusion PDE with Butler–Volmer boundary conditions. Solving this
  — the forward problem — is well understood. We are interested in
  the inverse.

- *Verification:* For each reaction type, the numerical solver is
  verified against known analytical solutions where they exist (e.g.
  reversible limit, irreversible limit, specific asymptotic regimes).
  These analytical results cover only edge cases, not the general
  parameter regime, but they confirm the correctness of the numerical
  implementation before it is used for inference. Mention this as a
  recurring element across chapters.

- *Why Bayesian:* Point estimates from optimisation do not capture
  parameter correlations or non-identifiability. A Bayesian approach
  gives the full posterior, quantifying what the data can and cannot
  tell us.

- *Why gradients:* MCMC sampling requires many evaluations of the
  forward map; gradient information accelerates both the burn-in
  (via optimisation-based initialisation) and the sampling itself
  (via HMC). Computing these gradients efficiently is a key technical
  challenge.

- *Kitchin et al. (2025) — "Beyond the fourth paradigm":* Positions
  differentiable programming as the foundation of a fifth paradigm in
  chemical engineering, beyond correlations, analytical fitting,
  numerical methods, and ML. Our work is a concrete instantiation of
  this vision in electrochemistry: we make a physics-based PDE solver
  fully differentiable and use the gradients for rigorous statistical
  inference.

- *Chen et al. (2026) — "Differentiable Electrochemistry":* Introduces
  end-to-end differentiable electrochemical simulators and demonstrates
  gradient-based optimisation for parameter discovery (Fe³⁺/Fe²⁺, HER,
  Li transport). Our work is complementary in two respects: (a) we take
  a Bayesian rather than optimisation-only approach, obtaining full
  posterior distributions with uncertainty quantification; (b) we
  compute gradients via a custom adjoint of the tridiagonal solve
  rather than taping all operations in the linear solver, which is
  both faster and more memory-efficient.

- *Scope:* The methodology is developed and validated on the simple
  electron-transfer reaction (A + e^- <=> B), then extended to a
  heterogeneous reaction (coupled redox + surface chemical step) and
  an adsorption reaction, demonstrating scalability to higher
  dimensions.

- *Tone:* Keep qualitative — no specific numerical results in the
  introduction. Each chapter presents its own results in context. The
  introduction should make the reader understand *why* each tool is
  needed and *what* they will gain from reading on.

- **TODO:** Return to finalise the introduction once results for the
  heterogeneous and adsorption reactions are complete. The introduction
  needs to state what was achieved for all three reaction types, so it
  cannot be written in final form until those chapters are done.

**Key references:**

- Kitchin, Alves & Laird (2025). "Beyond the fourth paradigm of
  modeling in chemical engineering." *Nat. Chem. Eng.* 2, 11–13.
- Chen, Huang, Rodríguez, Mistry & Viswanathan (2026). "Differentiable
  Electrochemistry: A Paradigm Characterizing Physical Laws in
  Electrochemical Systems." *ACS Energy Lett.*
- Compton & Banks — *Understanding Voltammetry* (standard reference for
  the forward problem and electrochemical background)
- Gavaghan et al. (2018) — Bayesian inference in electrochemistry
  (establishes the prior work on MCMC for voltammetric inference that
  this dissertation builds on)
- JAX (Bradbury et al.) — the AD framework used for implementation
- Standard references for HMC (Neal, Betancourt), MCMC
  (Hastings), adjoint methods (Kidger / Griewank & Walther)

---

## Chapter 2: Electrochemical Systems

**Status:** Written — largely complete, needs condensing

**Key message:** Formulate the forward problem for the single-electron
redox reaction A + e^- <=> B as a 1D diffusion PDE with Butler–Volmer
boundary conditions. Non-dimensionalise to obtain a system that can be
solved once and rescaled to any physical parameter set. Reduce to a
single species under equal diffusion coefficients.

**Assessment:**

- *Strengths:* Clear logical progression from physical setup through to
  the simplified dimensionless system. Good physical intuition
  (kinetic vs diffusional control, reversible limit). Notation
  carefully introduced. The equal-diffusion reduction is well motivated
  and the K_red/K_ox shorthand is a useful bridge to Chapter 3.

- *Weaknesses:* Several passages explain textbook material at length
  that the examiners will already know. The non-dimensionalisation
  derivations show intermediate algebra that could be stated as
  results. This is where word count can be recovered.

**Condensing targets (revised given examiner assumptions):**

Given that the markers know almost nothing about electrochemistry, the
CV description and peak-shape intuition should be kept at roughly
current length. The savings come from the algebra instead.

1. *Non-dimensionalisation algebra:* The explicit LHS/RHS substitution
   for the governing equation, scan rate, and current can each be
   replaced with "Substituting the scalings from Table 2.1 yields..."
   followed by the result. Applied mathematicians can verify this
   instantly. Estimated saving: ~150–200 words.

2. *Scan rate derivation:* Straightforward change of variables; state
   the result and note how σ relates to ν. Estimated saving: ~50 words.

3. *Current / flux recovery:* Same pattern — show result, not
   intermediate steps. Estimated saving: ~50–80 words.

4. *CV description:* **Keep.** The markers need this context. The
   kinetic-control -> diffusional-control transition is essential
   motivation for later chapters. Do not cut.

**Total estimated saving: ~250–330 words** (less than before, but the
right things are being kept)

**Keep at full length:**

- Butler–Volmer equation and its physical interpretation (central to
  the whole dissertation)
- Equal diffusion coefficient reduction (non-trivial, needed for
  Chapter 3)
- K_red / K_ox definitions (used extensively in numerics)
- The table of normalisations (compact, high reference value)

**Must mention:**

- Three-electrode cell setup (working, counter, reference)
- Fick's first and second laws
- Butler–Volmer kinetics with physical interpretation of each term
- Neumann boundary condition from combining Fick's first law with BV
- Mass conservation at electrode surface
- Initial and far-field boundary conditions (with Einstein
  displacement argument for domain truncation)
- Non-dimensionalisation scheme (table)
- Equal diffusion coefficient simplification -> single-species problem
- K_red, K_ox shorthand

**Key references:**

- Compton & Banks — *Understanding Voltammetry* (general
  electrochemistry, voltammetric techniques)
- Crank (1979) — *The Mathematics of Diffusion* (Fick's laws)
- Dickinson & Wain (2020) — Butler–Volmer equation
- Einstein (1905) — Brownian motion / diffusion length argument for
  domain truncation

---

## Chapter 3: Numerical Methods

**Status:** Written — good content, significant condensing possible

**Key message:** Discretise the dimensionless forward problem using an
implicit finite difference scheme on an exponentially expanding spatial
mesh. The tridiagonal system is solved at each timestep via the Thomas
algorithm. The forward solver is then viewed as a mapping 𝒫 from
parameters to predicted current, which is the central object for
inference. The effects of each Butler–Volmer parameter on the
voltammogram are illustrated, revealing correlations and
non-identifiability that motivate the Bayesian approach.

**Assessment:**

- *Strengths:* Clean logical flow from discretisation to the forward map
  abstraction. The exponentially expanding mesh is well motivated. The
  parameter effects section (3.4) is one of the strongest passages in
  the dissertation — it directly sets up the need for Bayesian
  inference by showing correlations and the non-identifiability of α
  in the reversible limit. The three-point flux formula is a nice
  detail. Defining 𝒫 formally is a good bridge to Chapter 4.

- *Weaknesses:* This chapter explains the most to an audience that
  needs it least. The markers are experts in finite differences,
  tridiagonal systems, and implicit time-stepping. Much of the
  derivation (interior coefficients, rearrangement into tridiagonal
  form, far-field row, Thomas algorithm description) is standard
  material that can be heavily condensed.

**Condensing targets (given examiner assumptions):**

The markers know FD methods deeply. They do *not* need:

1. *Backward Euler stencil derivation + rearrangement into tridiagonal
   form (Sections 3.2, 3.2.1):* State the scheme, give the
   coefficients α_i, β_i, γ_i directly, note it's implicit for
   unconditional stability. Cut the intermediate rearrangement.
   Estimated saving: ~100–150 words.

2. *Far-field boundary (Section 3.2.3):* "The Dirichlet condition is
   imposed directly" is sufficient — the explicit \beta=1, b=1 equations
   are unnecessary. Estimated saving: ~30–50 words.

3. *Thomas algorithm description (Section 3.2.4):* "The tridiagonal
   system is solved in O(n) by the Thomas algorithm" is one sentence.
   The current paragraph explaining what the diagonals are and that the
   interior coefficients are time-independent can be trimmed. Estimated
   saving: ~50–80 words.

4. *Temporal grid section (3.1.1):* Defining \delta T = T_max/m and
   T^k = k\deltaT is very standard. Could be one sentence. Estimated
   saving: ~30–50 words.

**Total estimated saving: ~210–330 words**

**Keep at full length:**

- Exponentially expanding spatial mesh — specific to electrochemistry,
  the markers won't have seen this motivation before
- Electrode boundary discretisation (how BV enters the first row) —
  this is the non-standard part of the tridiagonal system
- Three-point flux formula for current computation
- Forward map \mathcal{P} definition — conceptual bridge to Chapter 4
- **Section 3.4 (parameter effects) — do not cut a word.** This is
  the best motivation for the Bayesian chapter and shows real
  understanding of the interplay between mathematics and
  electrochemistry. The figures showing α insensitivity in the
  reversible regime are essential.

**Must mention:**

- Implicit (backward Euler) scheme for unconditional stability
- Exponentially expanding mesh with parameters h₀ and ω
- Non-uniform mesh central difference stencil
- Butler–Volmer boundary enters as time-dependent first row
- Three-point flux formula (higher order than BV discretisation)
- Forward map 𝒫: Φ -> 𝒥 as formal object
- Verification against analytical solutions in limiting cases (e.g.
  reversible limit, fully irreversible limit) to confirm correctness
  of the numerical implementation before using it for inference
- Effect of α: asymmetry between cathodic/anodic peaks
- Effect of K₀: symmetrising, sharpening, peak separation
- Effect of θ_f: pure translation along potential axis
- Parameter correlations and compensating errors
- Non-identifiability of α in the reversible limit (K₀ large)
- These observations motivate probabilistic inference -> Chapter 4

TODO: Add the reference to the analytical result here

**Key references:**

- Compton & Banks — *Understanding Voltammetry* (standard numerical
  approach to voltammetric simulation, expanding mesh)
- No additional references needed for standard FD material — the
  audience knows it

---

## Chapter 4: Bayesian Approach to the Inverse Problem

**Status:** Written — well-structured, moderate condensing possible

**Key message:** The inverse problem is ill-posed: noisy data, parameter
correlations, and non-identifiability mean that point estimates are
insufficient. We adopt a Bayesian approach, treating parameters as
random variables and seeking their posterior distribution. RWMH
sampling recovers the correct posterior but wastes substantial
computation on burn-in, motivating gradient-based methods.

**Assessment:**

- *Strengths:* Excellent narrative arc. The noise model -> Bayes ->
  likelihood -> MCMC -> RWMH progression is logical. The results
  section is the real payoff: histograms confirm recovery, the scatter
  plot showing burn-in waste is the single best motivation for
  Chapters 5–6. The final paragraph bridging to gradient-based methods
  is a perfect transition. The chapter makes the reader *want*
  gradients before they've been introduced.

- *Weaknesses:* The Bayesian derivation chain is shown in full detail
  (Bayes -> uniform prior -> posterior ∝ likelihood -> Gaussian noise ->
  product -> log -> drop constant). The markers know enough Bayesian
  statistics that most intermediate steps can be condensed. The
  general MH acceptance probability is stated and then immediately
  specialised to RWMH — the general form could be a single sentence.
  The Markov chain / ergodic theorem paragraph is context-setting that
  could be tightened.

**Condensing targets (given examiner assumptions):**

The markers know "a bit" about Bayesian statistics — they understand
priors, posteriors, and likelihoods but may not work with them daily.
So: state concepts clearly but don't derive what's standard.

1. *Posterior derivation (Section 4.2):* State Bayes' theorem in one
   equation, note the uniform prior makes posterior ∝ likelihood, then
   go straight to the log-likelihood ℓ(φ). Cut the intermediate steps
   showing the full likelihood product and the full log-likelihood
   before dropping the constant. The reader sees the noise model and
   the final ℓ(φ) — that's all they need. Estimated saving:
   ~100–150 words.

2. *Markov chain section (4.3.1):* The Markov property and ergodic
   theorem can be stated in one sentence each rather than a full
   paragraph. Estimated saving: ~50–80 words.

3. *General MH acceptance (4.3.3):* State the general form in one
   line, then move directly to the RWMH simplification. Cut the
   paragraph explaining proposal distributions abstractly (4.3.2) to
   a sentence — the RWMH Gaussian proposal is introduced immediately
   after anyway. Estimated saving: ~60–80 words.

4. *RWMH proposal covariance discussion:* The paragraph on tuning Σ
   and the 23% acceptance rate heuristic is useful context but could
   lose a sentence or two. Estimated saving: ~30–40 words.

**Total estimated saving: ~240–350 words**

**Keep at full length:**

- Noise model (eq 4.1) — defines the entire inference problem
- The log-likelihood ℓ(φ) — used in every subsequent chapter
- RWMH sampling results: histograms and scatter plot — these are
  results, not exposition, and they motivate everything that follows
- The burn-in discussion and its computational cost argument
- **Corner plot (biplot) of posterior samples:** Supervisor suggested
  including a pairs/corner plot to visualise pairwise correlations
  between parameters. For the 3-parameter electron-transfer case this
  is straightforward and should be included in Chapter 4. For the
  7-parameter (Chapter 7) and 9-parameter (Chapter 8) cases, the full
  corner plot becomes hard to read. Options: (a) show only the most
  interesting/correlated pairs, (b) show the full plot at reduced size
  as a supplementary figure, or (c) use a summary statistic like the
  correlation matrix instead. Decide once the higher-dimensional
  posteriors are available.
- **Corner plot / biplot:** Supervisor suggested including pairwise
  posterior correlation plots. The 3-parameter case is the natural
  place to introduce this — the full corner plot is readable and
  directly shows the correlations and non-identifiability discussed
  in Chapter 3. This sets a visual baseline that later chapters can
  refer back to.
- The final bridging paragraph to gradient-based methods

**Must mention:**

- **TODO:** The noise model currently uses a fixed ς². The standard in
  the field is to define the noise level as proportional to the peak
  current (maximum measured current), scaled down by some factor. This
  needs to be implemented and the chapter updated accordingly. The
  TODO in the tex source reflects this. Morris, 2013 says between 1% and 2% of the peak of the flux
- Bayes' theorem, uniform prior, posterior ∝ likelihood
- Log-likelihood ℓ(φ) as the working objective
- MCMC: construct chain whose stationary distribution is the posterior
- RWMH: symmetric Gaussian proposal, simplified acceptance ratio
- Tuning Σ, 23% acceptance rate heuristic (Gelman et al.)
- Results: marginal posteriors recover true parameters
- Key observation: burn-in is expensive because each likelihood
  evaluation requires a full forward solve
- Gradient-based optimisation could locate the mode -> initialise
  chain; gradient-based proposals (HMC) could improve sampling
- Both require differentiating through 𝒫 -> Chapter 5

**Key references:**

- Hastings (1970) — Metropolis–Hastings algorithm
- Gelman, Roberts & Gilks (1997) — optimal 23% acceptance rate
- Gavaghan et al. (2018) — Bayesian inference in electrochemistry
  (prior work this builds on)
- Gelman et al. (2013) — *Bayesian Data Analysis* (general reference)

---

## Chapter 5: Differentiating Through the Forward Solver

**Status:** Written — technically strong, some expository fat to trim

**Key message:** Computing the gradient of the log-likelihood with
respect to parameters requires differentiating through the sequence of
tridiagonal solves in the forward map. Naive AD taping is memory-
prohibitive (O(mn) across all timesteps). Instead, we exploit the
structure of the linear solve: the adjoint system A^T λ = x̄ is itself
tridiagonal, costing O(n) per timestep and requiring only A and x to
be stored. This is implemented as a custom VJP rule in JAX. The
gradient enables ADAM-based initialisation that outperforms CMA-ES in
speed, robustness, and stability.

**Assessment:**

- *Strengths:* The adjoint of the tridiagonal solve (Section 5.4) is
  the key technical contribution and is clearly presented. The
  contrast between naive taping and the structured adjoint is well
  drawn. The custom VJP implementation in JAX is a concrete detail
  that grounds the maths. The ADAM vs CMA-ES comparison (Section 5.6)
  is well-designed (equalised budgets, 32 starts, two noise levels)
  and gives an immediate payoff. The pentadiagonal extension shows
  generality. Discretise-then-optimise is justified concisely.

- *Weaknesses:* The reverse-mode AD explanation (Section 5.2) covers
  the chain rule and VJP concept at a level the markers won't need —
  they are applied mathematicians who will have encountered AD. The
  naive taping discussion (Section 5.3) makes a valid point but takes
  a paragraph where two sentences would suffice. Minor grammatical
  error in the opening ("gradient information is could be desirable").

**Condensing targets (revised — markers are NOT familiar with AD):**

Per supervisor guidance, the markers cannot be assumed to know
automatic differentiation. This means the reverse-mode AD section is
genuine exposition, not review. The savings here are smaller than
initially estimated.

1. *Reverse-mode AD section (5.2):* **Keep at current length or
   expand slightly.** This is new material for the audience. The chain
   rule decomposition, the distinction between forward and reverse
   mode, and the concept of VJPs all need proper explanation. Consider
   adding a brief remark on why finite differences and symbolic
   differentiation are inadequate (cost scales with number of
   parameters for finite differences; symbolic differentiation doesn't
   handle loops/branching in code). This would strengthen the
   motivation for AD.

2. *Naive taping discussion (5.3):* Now that AD itself is being
   explained, the taping discussion is part of the tutorial — the
   reader needs to understand what generic AD does before they can
   appreciate why the custom adjoint is better. **Keep**, but could
   tighten by ~30–50 words at sentence level.

3. *Opening paragraph:* Fix "is could be desirable" -> "could be
   desirable".

**Total estimated saving: ~30–50 words only** (this chapter needs its
current length given the audience)

**Additional note:** The AD exposition combined with the custom adjoint
derivation is a strength for the "Clarity" marking criterion (12–15:
"clarity of explanation is superb with ideas carefully, roundly
introduced; the effort made for the benefit of the reader is obvious").
Do not sacrifice this for word count savings.

**Keep at full length:**

- Discretise-then-optimise justification (already concise)
- The adjoint derivation: ā_b = λ^T from A^T λ = x̄, and
  Ā = -λx^T — this is the core contribution
- The observation that A^T is tridiagonal -> O(n) adjoint solve
- Memory comparison: custom adjoint stores A and x per timestep vs
  naive taping stores entire forward history
- Custom VJP rule in JAX: forward pass caches A, x; backward pass
  solves A^T λ = x̄
- ADAM vs CMA-ES results (Section 5.6) — these are results, keep
  in full
- Pentadiagonal extension — shows generality, already brief

**Must mention:**

- Discretise-then-optimise (not optimise-then-discretise) — computes
  exact gradient of the discrete computation
- Reverse-mode AD is the natural choice (scalar output)
- Naive taping of Thomas algorithm: O(mn) memory, prohibitive
- Adjoint system A^T λ = x̄ where x̄ is the incoming cotangent
- A^T tridiagonal -> adjoint solve is O(n)
- Ā = -λx^T for gradients w.r.t. matrix entries
- Only A and x need to be cached per timestep
- Custom VJP rule in JAX
- Extension to pentadiagonal (and wider banded) systems
- Initialisation results: ADAM vs CMA-ES, equalised budget,
  32 random starts, two noise levels
- ADAM converges faster, lower variance, stable at mode

**Key references:**

- Kidger (2021) — discretise-then-optimise vs optimise-then-discretise
- Kingma & Ba (2014) — ADAM optimiser
- Hansen & Ostermeier (2001) — CMA-ES
- JAX (Bradbury et al.) — AD framework
- Griewank & Walther (2008) — *Evaluating Derivatives* (AD reference)

---

## Chapter 6: Hamiltonian Monte Carlo

**Status:** Written — strong narrative, some condensing possible, two
gaps to address

**Key message:** HMC uses gradient information to construct long-range,
high-acceptance proposals by simulating Hamiltonian dynamics on the
posterior landscape. Combined with the adjoint-based gradients from
Chapter 5, this yields substantially more efficient sampling than RWMH
at equal computational cost. ChEES-HMC is used to adapt
hyperparameters, favouring shorter trajectories that reduce the number
of expensive forward+adjoint solves per sample.

**Assessment:**

- *Strengths:* The physical intuition (particle on a landscape, random
  kick, energy conservation) is the best way to introduce HMC to the
  audience. The potential energy = negative log-posterior connection
  ties back cleanly to Chapter 4. The leapfrog integrator and
  Metropolis correction are well presented. Algorithm 1 is a useful
  reference. The ESS comparison on a normalised time axis is the right
  metric and the right presentation — it's the culmination of the
  gradient story. ChEES-HMC is a good practical choice with clear
  justification for this setting.

- *Weaknesses:* The Hamiltonian setup (joint density factorisation,
  marginalisation recovering the target) is slightly over-explained.
  Hamilton's equations and leapfrog properties (symplectic,
  time-reversible) could be stated more tersely — the markers know
  ODEs. Two gaps need addressing: (1) the actual HMC posterior
  histograms are not shown (only ESS); (2) the Gelman–Rubin
  convergence diagnostic is not mentioned in any chapter.

**Condensing targets (given examiner assumptions):**

The markers know ODEs and numerical integration well. The Bayesian
context has been set up in Chapter 4. HMC itself is new to them, so
the physical intuition and algorithm should stay.

1. *Hamiltonian setup (6.1–6.1.2):* The factorisation P(q,p) ∝
   exp(-U)exp(-K) and the observation that marginalising over p
   recovers the target can each be one sentence rather than a full
   paragraph. The potential/kinetic energy definitions should stay.
   Estimated saving: ~80–100 words.

2. *Hamilton's equations + leapfrog (6.2):* State Hamilton's equations,
   state the leapfrog scheme, note it is symplectic and
   time-reversible. Cut any elaboration on what these properties mean
   — the markers know. Estimated saving: ~50–80 words.

3. *Mass matrix discussion (6.1.2):* The paragraph on M approximating
   the posterior covariance is useful context but could lose a sentence.
   Estimated saving: ~20–30 words.

4. *Metropolis correction (6.2.2):* The momentum negation and detailed
   balance explanation is good and should stay — HMC is new to the
   markers. Keep.

**Total estimated saving: ~150–210 words**

**Gaps to address:**

- **TODO: Include HMC posterior histograms.** The chapter currently
  shows only the ESS comparison, with a comment in the tex saying "I
  am not including the actual sampling results because I think it is
  clear that it will be fine with HMC." The markers will want to see
  that the posteriors are correct, not just that sampling is efficient.
  Even a single figure showing the marginals (analogous to the RWMH
  histograms in Chapter 4) would suffice.

- **TODO: Gelman–Rubin convergence diagnostic.** This is not mentioned
  anywhere in the dissertation. It should appear either here or in
  Chapter 4 (or both). The markers know enough Bayesian statistics to
  expect some convergence diagnostic beyond visual inspection.

- **TODO: Sampling space vs physical space.** There is a TODO comment
  in the tex about the distinction between parameters in the sampling
  space and in the physical space, and the importance of parameters
  being on the same scale for the uniform leapfrog step size. This
  should be addressed — it's a practical point that affects HMC
  performance and shows understanding of the method.

**Keep at full length:**

- Physical intuition opening (particle, kick, Hamiltonian landscape)
- Potential energy = -log posterior, kinetic energy = ½p^T M^{-1} p
- Leapfrog scheme (equations)
- Metropolis correction with momentum negation
- Algorithm box
- ESS comparison (normalised time axis, HMC vs RWMH)
- ChEES-HMC: adapts ε, L, M jointly; selects shorter trajectories
  than NUTS; warm-up serves as both burn-in and tuning

**Must mention:**

- Hamiltonian H(q,p) = U(q) + K(p)
- U(q) = -ℓ(q) (negative log-likelihood under uniform prior)
- p ~ N(0, M), mass matrix M
- Hamilton's equations, ∇_q U computed via adjoint (Chapter 5)
- Leapfrog integrator: symplectic, time-reversible
- Each leapfrog step = one gradient evaluation = one forward + adjoint
  solve
- Metropolis correction for discretisation error
- Momentum discarded after accept/reject, fresh draw each iteration
- ESS definition and comparison with RWMH at equal computational cost
- HMC achieves higher ESS across all parameters, largest gain for K₀
- ChEES-HMC over NUTS: shorter trajectories, fewer gradient
  evaluations per sample
- Warm-up phase: simultaneous adaptation + burn-in

**Key references:**

- Neal (2011) — MCMC using Hamiltonian dynamics
- Betancourt (2018) — conceptual introduction to HMC
- Hoffman et al. (2014) — NUTS
- Hoffman et al. (2021) — ChEES-HMC
- Gelman et al. (2013) / Geyer (2011) — ESS definition
- Gelman & Rubin — convergence diagnostic (once added)

---

## Chapter 7: Heterogeneous Reaction

**Status:** Early draft — equations written, prose and results incomplete

**Key message:** The methodology developed in Chapters 2–6 extends to a
chemically richer system: a heterogeneous reaction involving two redox
couples linked by a surface chemical step (B -> C with rate constant
K_het). This doubles the number of species and more than doubles the
parameter space (7 parameters). The chapter demonstrates that the
differentiable solver and HMC inference scale to this higher-dimensional
problem. It also serves as a stepping stone to Chapter 8 (adsorption),
of which this reaction is a simplification — the heterogeneous chemical
step here is instantaneous and irreversible, whereas the adsorption
treatment models it as a reversible surface process.

**Context and motivation:**

- *Electrochemical context:* This type of reaction (two electron
  transfers coupled by a heterogeneous chemical step at the electrode
  surface) arises in the electroreduction of halonitroaromatic
  compounds in aprotic media (Compton & Banks). Mention this briefly
  to ground the problem in real chemistry.

- *Relationship to adsorption (Chapter 8):* Frame this reaction as a
  simplification of the more general surface reaction treated in
  Chapter 8. In the heterogeneous reaction, species B is converted to
  C at the electrode surface at a fixed rate K_het — there is no
  explicit modelling of adsorbed surface concentrations. The
  adsorption chapter will relax this by modelling adsorption/desorption
  explicitly. This framing gives the reader a roadmap and justifies
  treating the heterogeneous case first.

**Numerical aspects — what changes from the electron-transfer reaction:**

- *Four species (A, B, C, D):* Four coupled diffusion equations instead
  of one (after the equal-diffusion simplification in Chapter 2, the
  electron-transfer reaction reduced to a single species; that
  reduction does not fully apply here because the chemical step
  couples B and C at the boundary).

- *Pentadiagonal system:* The four species with coupled boundary
  conditions produce a banded system. By reordering the unknowns —
  species A and B indexed from node N down to 1, species C and D
  indexed from node 1 up to N — the coupled boundary terms (which
  link B and C via K_het) sit adjacent in the banded matrix, reducing
  the bandwidth from nonadiagonal to pentadiagonal. This is a key
  implementation detail worth explaining clearly, possibly with a
  small schematic of the band structure. The markers (who know linear
  algebra well) will appreciate this.

- *Custom adjoint extends:* As noted in Chapter 5 (Section 5.5), the
  adjoint of a pentadiagonal system is also pentadiagonal. The custom
  VJP rule generalises directly. State this — it's the payoff of the
  earlier section.

- *No need to re-derive the full discretisation.* State the governing
  equations for the four species, the modified boundary conditions
  (which now include K_het coupling terms), and note that the
  discretisation follows the same implicit finite difference scheme as
  Chapter 3 with the spatial grid and time-stepping unchanged. Only
  describe what is *different*: the boundary condition structure, the
  species reordering trick, and the pentadiagonal solve.

**Sampling challenges:**

- *7 parameters:* α₁, K₀⁽¹⁾, θ_f⁽¹⁾, α₂, K₀⁽²⁾, θ_f⁽²⁾, K_het.
  This is more than double the electron-transfer case (3 parameters).
  The curse of dimensionality makes gradient-free methods
  increasingly expensive — this is where the advantage of HMC should
  become more pronounced.

- *Parameter correlations:* With two redox couples sharing the same
  voltammogram and a coupling constant, there may be richer
  correlation structure in the posterior. Discuss what the posteriors
  reveal (once results are available).

- *Non-identifiability questions:* Do the same issues arise as in the
  simple case (α insensitive in reversible limit)? Are there new
  degeneracies introduced by the coupling?

**Results (TODO):**

- Posterior distributions for all 7 parameters from HMC on synthetic
  data
- ESS comparison with RWMH at equal computational cost
- Demonstrate that the framework scales: the differentiable solver,
  custom adjoint, and HMC all work on the larger system
- **Corner plot:** Include the full 7×7 pairwise posterior plot, but
  highlight and discuss only the most interesting pairs (e.g.
  correlations between the two redox couples' parameters, K_het vs
  K₀ values). The full grid becomes harder to read at 7D — focus
  the discussion on what the correlations reveal about identifiability
  rather than describing every panel.
- Show the voltammogram fit (simulated vs synthetic data)
TODO: Add the reference to the analytical result here

**AC voltammetry:**

- **Brief mention only.** Supervisor suggested including a sinusoidal
  component in the applied potential waveform. Treatment: one
  paragraph noting that the framework handles AC voltammetry, state
  how the non-dimensionalisation extends (the sinusoidal perturbation
  enters through θ(T) in the boundary condition), and note that this
  does not change the solver structure. No full exposition, no
  dedicated figures unless space permits. If it adds value, one figure
  showing the AC voltammogram could be included, but this is low
  priority relative to the core results.

**Must mention:**

- Reaction scheme: A + e^- <=> B, B ->(k_het) C, C + e^- <=> D
- Electrochemical context (halonitroaromatic compounds)
- Relationship to adsorption chapter (this is the simplified case)
- Four coupled diffusion equations
- Modified boundary conditions with K_het coupling B and C
- Dimensionless K_het = k_het ε / D_A
- Total current = sum of contributions from both electrochemical
  reactions (flux of A + flux of C + K_het term)
- Species reordering trick -> pentadiagonal rather than nonadiagonal
- Custom pentadiagonal adjoint (callback to Chapter 5, Section 5.5)
- 7-parameter inference problem
- Initial/boundary conditions: only A present initially (C_A = 1,
  all others zero)
- Synthetic data only
- Verification of the numerical solver against analytical solutions
  in available limiting cases before using it for inference
- HMC results (TODO)
- Brief mention of AC voltammetry extension

**Key references:**

- Compton & Banks — *Understanding Voltammetry* (heterogeneous
  reaction mechanism, halonitroaromatic compounds)
- Chapter 3 of this dissertation (discretisation scheme — reference
  rather than re-derive)
- Chapter 5 of this dissertation (pentadiagonal adjoint extension)

---

## Chapter 8: Adsorption Reaction

**Status:** Chapter heading only — needs full writing

**Key message:** The adsorption reaction is the most complex system in
the dissertation: Langmuir adsorption/desorption with electron transfer
on the surface, coupled to diffusion in solution. It is a more rigorous
treatment of the surface chemistry that was simplified in Chapter 7.
The numerical challenge is that nonlinear terms in the surface coverage
ODE prevent a straightforward banded solve. We compare three
approaches (Newton's method, explicit linearisation, backward implicit
linearisation) and justify using the explicit scheme for inference on
the grounds of AD compatibility. The 9-parameter inference problem is
the highest-dimensional case in the dissertation.

**Context and motivation:**

- *Electrochemical context:* Adsorption reactions arise whenever
  electroactive species adsorb onto the electrode surface before
  undergoing electron transfer. This is common in electrocatalysis,
  biosensing, and corrosion studies. The Langmuir model
  (adsorption/desorption governed by surface coverage Γ with a
  maximum coverage Γ_max) is the standard first treatment.

- *Why solve this inverse problem:* In practice, adsorption parameters
  (adsorption/desorption rate constants, surface coverage capacity)
  are difficult to measure independently. Inferring them from
  voltammetric data alongside the kinetic parameters is valuable.
  The 9-parameter space makes gradient-free methods increasingly
  impractical — this is where the differentiable framework should
  show its strongest advantage.

- *Relationship to Chapter 7:* The heterogeneous reaction treated B -> C
  as an irreversible surface chemical step at a fixed rate K_het,
  without modelling surface concentrations explicitly. The adsorption
  reaction is a more rigorous treatment: species adsorb onto the
  surface (with finite coverage), react, and desorb. Chapter 7 is
  effectively the limiting case where adsorption/desorption is
  infinitely fast and surface coverage is not tracked.

**Reaction scheme:**

- A(solution) <=> A(ads) — Langmuir adsorption/desorption
- A(ads) + e^- -> B(ads) — electron transfer on surface
- B(ads) <=> B(solution) — product desorption

**Mathematical structure — what is different from previous chapters:**

- *Surface coverage ODE:* An ODE governs the surface coverage Γ(t),
  coupled to the diffusion PDE at the boundary only. The ODE contains
  nonlinear terms (products of surface coverage and solution
  concentration at x = 0).

- *Coupled ODE–PDE system:* The diffusion PDE in solution is the same
  as before (Fick's second law), but the boundary condition now
  involves Γ(t) from the surface ODE, and the surface ODE depends on
  the solution concentration at the electrode. This two-way coupling
  at the boundary is the key structural difference.

- *Nonlinearity:* The nonlinear terms (e.g. Langmuir adsorption rate
  ∝ c_A(0,t) · (Γ_max - Γ(t))) mean the system at each timestep is
  no longer a linear tridiagonal/banded system. Three approaches:

  1. **Newton's method:** Solve the full nonlinear system iteratively.
     Most accurate. BUT: convergence requires a variable number of
     iterations (while loop with data-dependent termination), which is
     incompatible with reverse-mode AD. The static computation graph
     required by AD cannot accommodate a data-dependent loop count.
     A fixed iteration count is possible but wasteful and offers no
     accuracy guarantee.

  2. **Explicit linearisation:** Evaluate all nonlinear terms at the
     previous timestep's values, so the system at each timestep is
     linear. Fully compatible with AD — one linear solve per timestep,
     no iteration. Accuracy depends on timestep size.

  3. **Backward implicit linearisation (Britz):** A more sophisticated
     linearisation from Britz's *Digital Simulation in
     Electrochemistry* that retains some implicit treatment of the
     nonlinear terms while keeping the system linear at each timestep.

- *Comparison of methods:* Compare all three for accuracy (against a
  high-resolution Newton reference solution) and for the computed
  current. Finding: explicit and backward implicit schemes produce
  similar results — no major difference in accuracy for the timestep
  sizes used. This justifies using the simpler explicit scheme.

- *AD compatibility argument:* Newton's method is incompatible with
  reverse-mode AD due to the variable iteration count (while loop).
  The explicit linearisation is fully differentiable. Strategy for
  inference: generate ground truth synthetic data with Newton's method
  (high accuracy, no need for AD), then solve the inverse problem
  with the explicit linearisation scheme (AD-compatible, sufficient
  accuracy). This is a pragmatic and defensible choice.

**Sampling:**

- *9 parameters:* The full parameter set for the adsorption reaction.
  Exact parameters TBD but will include: kinetic parameters for the
  electron transfer (α, K₀, θ_f), adsorption/desorption rate
  constants, and surface coverage parameters.

- *Highest-dimensional case:* 9 parameters vs 7 (heterogeneous) vs 3
  (electron transfer). The advantage of gradient-based sampling
  should be most pronounced here — gradient-free methods suffer most
  from the curse of dimensionality.

- *Posterior structure:* Expect richer correlations between kinetic
  and adsorption parameters. May reveal which parameters are
  identifiable from voltammetric data and which are not.

**Results (TODO):**

- Comparison of Newton vs explicit vs backward implicit linearisation
  (accuracy, current traces)
- Posterior distributions for all 9 parameters from HMC on synthetic
  data
- ESS comparison with RWMH at equal computational cost
- Voltammogram fit (simulated vs synthetic data)
- Discussion of parameter identifiability and correlations in the
  9D posterior
- **Corner plot:** At 9 parameters the full 36-panel corner plot is
  difficult to interpret. Consider showing only selected pairwise
  marginals that reveal the most interesting correlations (e.g.
  adsorption rate vs K₀, surface coverage parameters vs α). Reference
  the full 3D corner plot from Chapter 4 and the 7D plot from
  Chapter 7 to show how the correlation structure evolves with
  increasing dimensionality.

**Writing approach:**

- Open with electrochemical context and motivation (why adsorption
  matters, where it arises)
- Frame as more rigorous treatment of Chapter 7's surface chemistry
- State the reaction scheme and governing equations (ODE + PDE)
- Focus on what is *different* from previous chapters: the nonlinear
  coupling, the three solver approaches, the AD compatibility issue
- Do NOT re-derive the finite difference discretisation — reference
  Chapter 3 and state only the modifications
- Present the solver comparison results
- Then the inference results (once available)

**Must mention:**

- Langmuir adsorption/desorption scheme
- Surface coverage ODE coupled to PDE at boundary
- Nonlinear terms prevent direct banded solve
- Three approaches: Newton, explicit linearisation, Britz backward
  implicit
- Newton incompatible with reverse-mode AD (while loop /
  data-dependent termination)
- Explicit linearisation: AD-compatible, sufficient accuracy
- Strategy: ground truth from Newton, inference with explicit scheme
- 9-parameter inference — highest dimensional case
- Comparison of linearisation schemes (accuracy results)
- Verification of the numerical solver against analytical solutions
  in available limiting cases (edge cases only — analytical results
  do not cover the general nonlinear regime)
- HMC posteriors and ESS (TODO)
- Relationship to Chapter 7 (this is the rigorous version)

**Key references:**

- Britz — *Digital Simulation in Electrochemistry* (backward implicit
  linearisation schemes)
- Compton & Banks — *Understanding Voltammetry* (adsorption reactions,
  Langmuir model)
- Chapter 3 of this dissertation (discretisation — reference, don't
  re-derive)

---

## Chapter 9: Conclusion

**Status:** Chapter heading only — write last, once all results are in

**Key message:** Bring the arc together: we built a differentiable PDE
solver, used it for Bayesian inference via HMC, and showed it scales to
complex reactions. Then look forward: what would we do differently,
what remains open, what are the longer-term implications.

**Structure (per Ehrenberg and marking criteria):**

1. *Summary of contributions (~1 paragraph):* Pull the thread from
   the forward problem through to the final results. State what was
   achieved for each reaction type. Keep brief — the reader has just
   read the dissertation. This is not a re-exposition, it's a
   reminder of the punchline.

2. *Further work / limitations:* This is where the conclusion earns
   its marks. Ehrenberg: "What objectives did we not manage to cover?
   What might be done next?" The marking criteria for Coherence (8–10)
   reward "conclusions well presented." Candidates for discussion:

   - *Experimental data:* All results are on synthetic data. Applying
     the framework to real experimental voltammograms is the natural
     next step. Discuss what additional challenges arise (model
     misspecification, unmodelled effects like capacitive current,
     ohmic drop, electrode roughness).

   - *Noise model:* Currently fixed \varsigma^2. The standard in the field is
     noise proportional to peak current. If this has been implemented
     by the time of writing, summarise; if not, note it as a
     limitation and future improvement.

   - *Unknown noise variance:* Could treat ς² as an additional
     parameter to infer (hierarchical Bayesian model). This is a
     natural extension.

   - *Alternative kinetic models:* The dissertation uses Butler–Volmer
     throughout. Marcus–Hush–Chidsey kinetics (as in Chen et al.
     2026) could be incorporated into the differentiable solver.
     Model comparison / selection between BV and MHC is a natural
     Bayesian application.

   - *Higher-order spatial discretisations:* The pentadiagonal adjoint
     (Chapter 5, Section 5.5) was introduced but not fully exploited.
     Higher-order schemes could improve accuracy for the same grid
     size.

   - *Microelectrodes / 2D geometry:* The dissertation assumes a
     macro-electrode (1D diffusion). Microelectrodes require 2D
     diffusion, which changes the solver structure significantly.

   - *NUTS / other adaptive HMC variants:* ChEES-HMC was used; NUTS
     is the standard alternative. A comparison could be informative. MEADs or MCLMC

   - *Neural operator emulator:* The finite difference solver computes
     the full spatiotemporal concentration field C(X, T) at every
     spatial node, but the inference only uses the surface flux J(T)
     — a quantity determined by a few nodes near X = 0. This is
     computationally wasteful. A natural extension is to train a
     neural operator (e.g. DeepONet, Fourier Neural Operator) to
     learn the mapping 𝒫 directly: from the known time-dependent
     inputs (K_red(T), K_ox(T), which are determined entirely by the
     parameters and the applied potential waveform) to the flux
     waveform J(T). Training data can be generated cheaply using the
     finite difference solver. The resulting emulator would be
     differentiable by construction (it's a neural network), orders
     of magnitude faster per evaluation, and immediately compatible
     with HMC. This replaces the PDE solve entirely at inference
     time while retaining the physics through the training data.

   - *Scaling to even higher dimensions:* The adsorption reaction has
     9 parameters. Real electrochemical systems (e.g. multi-step
     mechanisms with many intermediates) could have many more. How
     does the approach scale?

3. *Broader implications (~1–2 sentences):* Connect to the Kitchin
   "fifth paradigm" framing from the introduction. This work is a
   concrete demonstration that differentiable simulation + Bayesian
   inference is a viable and efficient approach for electrochemical
   inverse problems, and the methodology is not specific to the
   reactions studied here.

**TODO:** Cannot be written in final form until Chapters 7 and 8
results are complete. Draft the structure and further-work items now;
fill in the summary once results are available.

**TODO:**  A note on Kinetic zones that is mentioned in (Gavaghan, 2017)

**Must mention:**

- Summary of the three contributions (solver, Bayesian inference,
  complex reactions)
- What worked well
- Limitations (synthetic data only, fixed noise model, BV kinetics
  only)
- Further work candidates (see list above — select the most
  compelling 3–4, don't list everything)
- Connection back to the introduction's framing (Kitchin fifth
  paradigm, complementarity with Chen et al.)

**Key references:**

- Kitchin et al. (2025) — callback to introduction framing
- Chen et al. (2026) — callback to introduction positioning
- Any references needed for specific further-work items (e.g.
  Marcus–Hush–Chidsey if mentioned)
