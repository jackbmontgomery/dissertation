from time import perf_counter

import jax.numpy as jnp
import pytest

from src.fdm import (
    SecondOrderECirreFDMSolverBackwardImplicit,
    SecondOrderECirreFDMSolverNewton,
)
from src.params import SecondOrderECirreMechanismFDMParams
from src.voltammetry import CyclicDC


@pytest.fixture(scope="module")
def ec_irre_reaction():
    voltammetry = CyclicDC()
    params = SecondOrderECirreMechanismFDMParams(
        alpha=jnp.array(1.0),
        K0=jnp.array(10000.0),
        Kplus=jnp.array(1000000.0),
        Kminus=jnp.array(10.0),
        dB=jnp.array(1.0),
        dY=jnp.array(1.0),
        dZ=jnp.array(1.0),
        E0=jnp.array(0.0),
    )
    return dict(params=params, voltammetry=voltammetry)


def test_agreement(ec_irre_reaction):
    h = 1e-3
    dtheta = 5e-2

    voltammetry = ec_irre_reaction["voltammetry"]
    params = ec_irre_reaction["params"]

    newton_solver = SecondOrderECirreFDMSolverNewton(voltammetry, h=h, dtheta=dtheta)

    backward_solver = SecondOrderECirreFDMSolverBackwardImplicit(
        voltammetry, h=h, dtheta=dtheta
    )

    current_backward = backward_solver.solve(params)
    current_newton = newton_solver.solve(params)

    mse = jnp.mean(jnp.square(current_backward - current_newton))

    assert mse < 1e-4
