from time import perf_counter

import jax.numpy as jnp
import pytest

from src.fdm import SecondOrderECirreFDMSolverExplicitApprox
from src.params import SecondOrderECirreMechanismFDMParams
from src.voltammetry import LinearSweepDC


@pytest.fixture(scope="module")
def ec_irre_reaction():
    params = SecondOrderECirreMechanismFDMParams(
        alpha=jnp.array(1.0),
        K0=jnp.array(1000.0),
        Kminus=jnp.array(1.0),
        Kplus=jnp.array(1000.0),
        E0=jnp.array(0.0),
        dB=jnp.array(1.0),
        dY=jnp.array(1.0),
        dZ=jnp.array(1.0),
    )
    return dict(params=params)


def test_performance(ec_irre_reaction):
    h = 1e-2
    dtheta = 5e-2

    voltammetry = LinearSweepDC()

    fdm_solver = SecondOrderECirreFDMSolverExplicitApprox(voltammetry, h, dtheta)

    params = ec_irre_reaction["params"]

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
