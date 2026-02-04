from time import perf_counter

import jax
import jax.numpy as jnp
import pytest

from src.fdm import MacroElectrodeFDMSolver, MicroElectrodeFDMSolver
from src.params import ElectrodeKineticsParameters
from src.voltammetry import LinearSweepDC


@pytest.fixture(scope="module")
def macroband():
    h = 1e-2
    dtheta = 5e-2

    voltammetry = LinearSweepDC()
    fdm_solver = MacroElectrodeFDMSolver(voltammetry, h, dtheta)

    params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.7), kappa=jnp.array(1.0), epsilon=jnp.array(0.0)
    )

    out = fdm_solver.solve(params)
    out.block_until_ready()

    return dict(fdm_solver=fdm_solver, params=params, sigma=voltammetry.sigma)


def test_macroband_accuracy(macroband):
    assert jnp.ones(1).dtype == jnp.float32, "This should be run with float32"
    fdm_solver = macroband["fdm_solver"]
    params = macroband["params"]
    sigma = macroband["sigma"]

    current = fdm_solver.solve(params)

    max_current_estimate = jnp.min(current)

    max_current = (
        -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(sigma)
    )  # Randles-Sevcik Equation

    assert pytest.approx(max_current_estimate, rel=0.02) == max_current


def test_macroband_performance(macroband):
    assert jnp.ones(1).dtype == jnp.float32, "This should be run with float32"
    fdm_solver = macroband["fdm_solver"]
    params = macroband["params"]

    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        out = fdm_solver.solve(params)
        out.block_until_ready()
        times.append(perf_counter() - t0)

    best_time = min(times)

    budget_s = 0.075
    assert best_time < budget_s, (
        f"Runtime {best_time:.3f}s exceeds budget {budget_s:.3f}s"
    )


@pytest.fixture(scope="module")
def microband():
    jax.config.update("jax_enable_x64", True)

    sigma = 10000.0
    h0 = 1e-4
    omega = 1.1
    dtheta = 0.05

    voltammetry = LinearSweepDC(sigma=sigma)
    fdm_solver = MicroElectrodeFDMSolver(voltammetry, h0, omega, dtheta)

    params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.7), kappa=jnp.array(1000.0), epsilon=jnp.array(0.0)
    )

    out = fdm_solver.solve(params)
    out.block_until_ready()

    return dict(
        fdm_solver=fdm_solver,
        params=params,
        sigma=sigma,
    )


def test_microband_accuracy(microband):
    fdm_solver = microband["fdm_solver"]
    params = microband["params"]
    sigma = microband["sigma"]

    current = fdm_solver.solve(params)
    current.block_until_ready()
    max_current_estimate = float(jnp.min(current))

    p = jnp.sqrt(sigma)
    max_current = 0.439 * p + 0.713 * p**0.108 + (0.614 * p) / (1 + 10.9 * p**2)
    expected = -1.0 * float(max_current)

    assert pytest.approx(expected, rel=0.02) == max_current_estimate


def test_microband_performance(microband):
    fdm_solver = microband["fdm_solver"]
    params = microband["params"]

    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        out = fdm_solver.solve(params)
        out.block_until_ready()
        times.append(perf_counter() - t0)

    best_time = min(times)

    budget_s = 0.4
    assert best_time < budget_s, (
        f"Runtime {best_time:.3f}s exceeds budget {budget_s:.3f}s"
    )
