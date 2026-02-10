from time import perf_counter

import jax.numpy as jnp
import pytest

from src.fdm import UnEReactionFDMSolver
from src.params import ElectrodeKineticsParameters2
from src.voltammetry import LinearSweepDC


@pytest.fixture(scope="module")
def e_reaction():
    h = 1e-2
    dtheta = 5e-2

    voltammetry = LinearSweepDC()
    fdm_solver = UnEReactionFDMSolver(voltammetry, h, dtheta)

    params = ElectrodeKineticsParameters2(
        alpha=jnp.array(0.7),
        kappa=jnp.array(1.0),
        epsilon=jnp.array(0.0),
        dB=jnp.array(1.0),
    )

    out = fdm_solver.solve(params)
    out.block_until_ready()

    return dict(fdm_solver=fdm_solver, params=params, sigma=voltammetry.sigma)


def test_accuracy(e_reaction):
    assert jnp.ones(1).dtype == jnp.float32, "This should be run with float32"
    fdm_solver = e_reaction["fdm_solver"]
    params = e_reaction["params"]
    sigma = e_reaction["sigma"]

    current = fdm_solver.solve(params)

    max_current_estimate = jnp.min(current)

    max_current = (
        -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(sigma)
    )  # Randles-Sevcik Equation

    assert pytest.approx(max_current, rel=0.02) == max_current_estimate


def test_performance(e_reaction):
    assert jnp.ones(1).dtype == jnp.float32, "This should be run with float32"
    fdm_solver = e_reaction["fdm_solver"]
    params = e_reaction["params"]

    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        out = fdm_solver.solve(params)
        out.block_until_ready()
        times.append(perf_counter() - t0)

    best_time = min(times)

    budget_s = 0.1
    assert best_time < budget_s, (
        f"Runtime {best_time:.3f}s exceeds budget {budget_s:.3f}s"
    )
