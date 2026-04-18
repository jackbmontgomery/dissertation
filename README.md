# Differentiable Simulation for the Electrochemical Inverse Problem

OMMS Dissertation -- Jack Montgomery, 2026

---

## What is this?

Voltammetry experiments measure current response as electrode potential is swept
-- the shape of that current encodes electrochemical parameters (rate constants,
transfer coefficients, formal potentials). Recovering those parameters from a
noisy current trace is the **electrochemical inverse problem**.

This project builds a full pipeline to solve it:

1. **Differentiable PDE solver** -- finite difference simulation of coupled
   diffusion-reaction PDEs on an exponentially expanding mesh, with exact
   gradients via a custom adjoint
2. **Gradient-based optimisation** -- fast mode-finding to initialise inference
   (ADAM vs CMA-ES)
3. **Bayesian inference** -- full posterior distributions over parameters via
   both Random-Walk Metropolis-Hastings (RWMH) and No-U-Turn Sampler (NUTS),
   powered by [BlackJAX](https://github.com/blackjax-devs/blackjax)

Applied to three reaction mechanisms of increasing complexity: simple electron
transfer (3 params), heterogeneous ECE reaction (7 params), and Langmuir
adsorption (9 params).

---

## Key Features

### Custom VJP Rule for the Tridiagonal Solve

Naively differentiating through a PDE solver via automatic differentiation tapes
every intermediate state -- O(mn) memory for m timesteps of size n. Instead, the
solver implements a **custom reverse-mode (VJP) rule** exploiting the adjoint
structure:

```
A x = b  ->  A^T lam = x_bar  (adjoint is also tridiagonal)
A_bar = -lam x^T
```

This reduces memory to O(n) per timestep, caching only `A` and `x`. The same
adjoint pattern extends to pentadiagonal systems for the coupled ECE reaction.
Registered directly in JAX via `jax.custom_vjp`.

### JAX-Native Throughout

The entire solver stack -- mesh construction, finite difference assembly,
time-stepping, likelihood evaluation, and sampling -- runs under JAX. This means:

- `jit` compilation of full forward + adjoint passes
- `vmap` over parameter batches for parallel chain initialisation
- Seamless interop with BlackJAX's NUTS and RWMH kernels

### C++ Banded Linear Solvers via FFI

The tridiagonal and pentadiagonal solves are backed by **C++ implementations**
called via JAX's Foreign Function Interface (FFI). Thomas algorithm for
tridiagonal, banded LU for pentadiagonal -- both compiled as shared libraries
(`tri_ffi.so`, `penta_ffi.so`) and registered as JAX primitives. Custom VJP
rules are defined at the JAX level, delegating the forward solve to C++.

### Pentadiagonal Reordering Trick

The ECE reaction couples four diffusion equations. A naive spatial ordering
produces a nonadiagonal system. By **reversing the index direction** for species
B and C, all coupling terms become adjacent -- collapsing to a pentadiagonal
system. This halves solver complexity and lets the same banded adjoint machinery
apply directly.

### Full Bayesian Workflow

Rather than point estimates, the framework produces **posterior distributions**
over all parameters:

- Latin Hypercube Sampling for diverse chain initialisation
- ADAM optimisation to find high-density regions before sampling
- Window adaptation for mass matrix and step size tuning
- Equal wall-time comparison between NUTS and RWMH

NUTS advantage over RWMH grows with parameter dimension (3 -> 7 -> 9 params),
consistent with the geometry-exploiting proposal structure.

---

## Structure

```
src/
    fdm/                  # Finite difference solvers (electron, heterogeneous, adsorption)
    linear_solvers/       # Tridiagonal + pentadiagonal solvers with custom VJPs
        ffi/              # C++ FFI backends (CMake build)
    reaction/             # Reaction mechanism definitions and parameters
    sampling/             # NUTS / RWMH inference wrappers (BlackJAX)
    optimisers/           # ADAM and CMA-ES optimisation
    diagnostics.py        # ESS, Gelman-Rubin, convergence tools
    plotting.py           # Corner plots, current traces, diagnostics

run_scripts/
    sampling.py           # Run full inference pipeline
    optimisation.py       # Run optimisation experiment

manuscript/               # LaTeX dissertation source
```

---

## Setup

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for dependency
management.

```bash
git clone <repo>
cd dissertation

# Install dependencies
uv sync

# Build C++ FFI solvers
cd src/linear_solvers/ffi
cmake -B build && cmake --build build
cd ../../..
```

---

## Running

### Inference (sampling)

```bash
uv run python run_scripts/sampling.py --name {e,h,a} [--seed INT] [--save BOOL]
```

Arguments:

- `--name` -- reaction: `e` (electron transfer), `h` (heterogeneous ECE), `a` (adsorption)
- `--seed` -- random seed (default: 0); seed 1 selects alternate config (reversible electron / AC voltammetry for heterogeneous)
- `--save` -- save results to disk (default: True)

### Optimisation experiment

```bash
uv run python run_scripts/optimisation.py --name {e,h,a} [--noise FLOAT] [--seed INT] [--save BOOL]
```

Arguments:

- `--name` -- reaction: `e`, `h`, or `a` (same as above)
- `--noise` -- noise level as fraction of peak current (default: 0.02)
- `--seed` -- random seed (default: 0)
- `--save` -- save results to disk (default: True)

Runs ADAM vs CMA-ES on a fixed forward-solve budget with 32 random starts.

### Tests

```bash
uv run pytest tests/
```

Includes verification against Randles-Sevcik analytical peak currents
(reversible and irreversible limits).

---

## Dependencies

| Package                  | Role                     |
| ------------------------ | ------------------------ |
| `jax`                    | Array ops, JIT, autodiff |
| `blackjax`               | NUTS and RWMH kernels    |
| `optax`                  | ADAM optimiser           |
| `equinox`                | PyTree utilities         |
| `evosax`                 | CMA-ES                   |
| `matplotlib` / `seaborn` | Plotting                 |
