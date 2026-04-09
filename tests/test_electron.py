import statistics
from time import perf_counter

import jax.numpy as jnp
import pytest

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC


@pytest.fixture(scope="module")
def e_reaction():
    voltammetry = CyclicDC()
    fdm_solver = ElectronReactionFDSolver(voltammetry)

    params = ElectronReactionParams(
        alpha=jnp.array(0.7),
        K0=jnp.array(1.0),
        thetaf=jnp.array(0.0),
    )

    _, out = fdm_solver.solve(params)
    out.block_until_ready()

    return dict(fdm_solver=fdm_solver, params=params, sigma=voltammetry.sigma)


def test_peak_current_value(e_reaction):
    fdm_solver = e_reaction["fdm_solver"]
    params = e_reaction["params"]
    sigma = e_reaction["sigma"]

    _, current = fdm_solver.solve(params)

    max_current_estimate = jnp.min(current)

    max_current = -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(sigma)

    assert pytest.approx(max_current, rel=0.02) == max_current_estimate


def test_peak_current_position(e_reaction):
    fdm_solver = e_reaction["fdm_solver"]
    params = e_reaction["params"]
    sigma = e_reaction["sigma"]

    _, current = fdm_solver.solve(params)

    peak_idx = jnp.argmin(current)

    peak_position_numerical = fdm_solver.applied_potentials[peak_idx]

    peak_position_analytical = (
        jnp.log(params.K0 / jnp.sqrt(params.alpha * sigma)) - 0.78
    ) / params.alpha

    assert pytest.approx(peak_position_numerical, rel=0.02) == peak_position_analytical


def test_performance(e_reaction):
    fdm_solver = e_reaction["fdm_solver"]
    params = e_reaction["params"]

    n_runs = 10
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        _, out = fdm_solver.solve(params)
        out.block_until_ready()
        times.append(perf_counter() - t0)

    best_time = min(times)

    print(
        f"Best time: {best_time:.4f}",
        f"Average time: {statistics.mean(times):.4f}",
    )

    budget_s = 0.038
    assert best_time < budget_s, (
        f"Runtime {best_time:.3f}s exceeds budget {budget_s:.3f}s"
    )
