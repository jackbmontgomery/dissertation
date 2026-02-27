import statistics
from time import perf_counter

import jax.numpy as jnp
import pytest

from src.fdm import HeterogeneousReactionFDSolver
from src.params import HeterogenousReactionParams
from src.voltammetry import CyclicDC


@pytest.fixture(scope="module")
def heterogeneous_reaction():
    voltammetry = CyclicDC()
    fdm_solver = HeterogeneousReactionFDSolver(voltammetry)

    params = HeterogenousReactionParams(
        alpha_1=jnp.array(0.7),
        K0_1=jnp.array(1.0),
        Ef_1=jnp.array(0.0),
        alpha_2=jnp.array(0.0),
        K0_2=jnp.array(0.0),
        Ef_2=jnp.array(0.0),
        dB=jnp.array(1.0),
        dC=jnp.array(1.0),
        dD=jnp.array(1.0),
        K_het=jnp.array(0.0),
    )

    out = fdm_solver.solve(params)
    out.block_until_ready()

    return dict(fdm_solver=fdm_solver, params=params, sigma=voltammetry.sigma)


def test_peak_current_value(heterogeneous_reaction):
    fdm_solver = heterogeneous_reaction["fdm_solver"]
    params = heterogeneous_reaction["params"]
    sigma = heterogeneous_reaction["sigma"]

    current = fdm_solver.solve(params)

    max_current_estimate = jnp.min(current)

    max_current = -0.496 * jnp.sqrt(params.alpha1) * jnp.sqrt(sigma)

    assert pytest.approx(max_current, rel=0.02) == max_current_estimate


def test_peak_current_position(heterogeneous_reaction):
    fdm_solver = heterogeneous_reaction["fdm_solver"]
    params = heterogeneous_reaction["params"]
    sigma = heterogeneous_reaction["sigma"]

    current = fdm_solver.solve(params)

    peak_idx = jnp.argmin(current)

    peak_position_numerical = fdm_solver.applied_potentials[peak_idx]

    peak_position_analytical = (
        jnp.log(params.K1_0 / jnp.sqrt(params.alpha1 * sigma)) - 0.78
    ) / params.alpha1

    assert pytest.approx(peak_position_numerical, rel=0.02) == peak_position_analytical


def test_performance(heterogeneous_reaction):
    fdm_solver = heterogeneous_reaction["fdm_solver"]
    params = heterogeneous_reaction["params"]

    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        out = fdm_solver.solve(params)
        out.block_until_ready()
        times.append(perf_counter() - t0)

    best_time = min(times)

    print(
        f"Best time: {best_time:.4f}",
        f"Average time: {statistics.mean(times):.4f}",
    )

    budget_s = 0.06
    assert best_time < budget_s, (
        f"Runtime {best_time:.3f}s exceeds budget {budget_s:.3f}s"
    )
