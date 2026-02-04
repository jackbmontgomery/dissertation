import jax.numpy as jnp

from src.fdm import MacroElectrodeFDMSolver, MicroElectrodeFDMSolver
from src.params import ElectrodeKineticsParameters
from src.voltammetry import LinearSweepDC


def test_butler_volmer_model():
    params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.7), kappa=jnp.array(1.0), epsilon=jnp.array(0.0)
    )
    voltammetry = LinearSweepDC()

    fdm_solver = MacroElectrodeFDMSolver(voltammetry, 1e-2, 5e-2)

    current = fdm_solver.solve(params)

    max_current_estimate = jnp.min(current)

    max_current = (
        -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(voltammetry.sigma)
    )  # Randles-Sevcik Equation

    assert jnp.isclose(max_current_estimate, max_current, rtol=0.02)


def test_microband():
    sigma = 20000.0
    h0 = 1e-4
    omega = 1.1
    dtheta = 0.05

    voltammetry = LinearSweepDC(sigma=sigma)
    fdm_solver = MicroElectrodeFDMSolver(voltammetry, h0, omega, dtheta)

    params = ElectrodeKineticsParameters(
        alpha=jnp.array(0.7), kappa=jnp.array(1000.0), epsilon=jnp.array(0.0)
    )

    current = fdm_solver.solve(params)
    max_current_estimate = jnp.min(current)

    p = jnp.sqrt(sigma)
    max_current = 0.439 * p + 0.713 * p**0.108 + (0.614 * p) / (1 + 10.9 * p**2)
    assert jnp.isclose(max_current_estimate, -1 * max_current, rtol=0.02)
