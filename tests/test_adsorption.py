import jax.numpy as jnp
import pytest

from src.fdm import (
    AdsorptionReactionBackwardImplicitFDSolver,
    AdsorptionReactionExplicitFDSolver,
    AdsorptionReactionNewtonFDSolver,
)
from src.params import AdsorptionReactionParams
from src.voltammetry import CyclicDC


@pytest.fixture(params=["newton", "backward", "explicit"])
def adsorption_reaction(request):
    voltammetry = CyclicDC(theta_i=25.0, theta_v=-25.0, sigma=40)
    if request.param == "newton":
        fdm_solver = AdsorptionReactionNewtonFDSolver(voltammetry)
    if request.param == "backward":
        fdm_solver = AdsorptionReactionBackwardImplicitFDSolver(voltammetry)
    else:
        fdm_solver = AdsorptionReactionExplicitFDSolver(voltammetry)
    return dict(fdm_solver=fdm_solver, sigma=voltammetry.sigma)


def test_monolayer_analytical(adsorption_reaction):
    params = AdsorptionReactionParams(
        alpha_sol=jnp.array(0.5),
        K0_sol=jnp.array(0.0),
        thetaf_sol=jnp.array(0.0),
        alpha_ads=jnp.array(0.5),
        K0_ads=jnp.array(1e6),
        K_A_ads=jnp.array(10e3),
        K_A_des=jnp.array(1e-3),
        K_B_ads=jnp.array(1.0),
        K_B_des=jnp.array(1e-3),
    )
    fdm_solver = adsorption_reaction["fdm_solver"]
    sigma = adsorption_reaction["sigma"]
    current = fdm_solver.solve(params)
    analytical_flux = (
        -sigma
        * jnp.exp(-(fdm_solver.applied_potentials - params.thetaf_ads))
        / ((1.0 + jnp.exp(-(fdm_solver.applied_potentials - params.thetaf_ads))) ** 2)
    )
    assert jnp.mean(analytical_flux - current) < 0.0005
