import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from src.fdm import AdsorptionReactionNewtonFDSolver
from src.params import AdsorptionReactionParams
from src.voltammetry import CyclicDC, LinearSweepDC


@pytest.fixture
def adsorption_reaction():
    voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=40)
    fdm_solver = AdsorptionReactionNewtonFDSolver(
        voltammetry, h0=1e-6, dtheta=1e-1, omega=1.1
    )
    return dict(fdm_solver=fdm_solver, sigma=voltammetry.sigma)


def test_monolayer_analytical(adsorption_reaction):
    params = AdsorptionReactionParams(
        alpha_sol=jnp.array(0.5),
        K0_sol=jnp.array(0.0),
        Ef_sol=jnp.array(0.0),
        alpha_ads=jnp.array(0.5),
        K0_ads=jnp.array(1e6),
        K_A_ads=jnp.array(10e3),
        K_A_des=jnp.array(1e-3),
        K_B_ads=jnp.array(1.0),
        K_B_des=jnp.array(1e-3),
        dB=jnp.array(1.0),
    )
    fdm_solver = adsorption_reaction["fdm_solver"]
    sigma = adsorption_reaction["sigma"]

    current = fdm_solver.solve(params)

    analytical_flux = (
        -sigma
        * jnp.exp(-(fdm_solver.applied_potentials - params.Ef_ads))
        / ((1.0 + jnp.exp(-(fdm_solver.applied_potentials - params.Ef_ads))) ** 2)
    )

    assert jnp.mean(analytical_flux - current) < 0.0005
