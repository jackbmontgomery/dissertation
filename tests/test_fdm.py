import jax.numpy as jnp
from jax import vmap

from src.experiment import CyclicMacroBand1D
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    discretise_experiment,
)
from src.pde_parameters import ButlerVolmerPhysicalParameters
from src.simulate import create_fdm_current_simulator


def test_butler_volmer_model():
    experiment = CyclicMacroBand1D()
    dx = 5e-2
    T, X = discretise_experiment(experiment, dx=dx)

    potentials = vmap(experiment.potential)(T)

    fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X)
    c_init = jnp.ones_like(X)

    simulate_current = create_fdm_current_simulator(
        c_init, potentials, fdm_discretisation, dx
    )
    params = ButlerVolmerPhysicalParameters(
        alpha=jnp.array(0.75), kappa0=jnp.array(1.0)
    )

    current_sim = simulate_current(params)
    max_current_estimate = jnp.min(current_sim)

    max_current = (
        -0.496 * jnp.sqrt(params.alpha) * jnp.sqrt(experiment.sigma)
    )  # Randles-Sevcik Equation

    tolerance = -0.05

    lower_max_current = max_current - tolerance * max_current
    upper_max_current = max_current + tolerance * max_current

    assert lower_max_current <= max_current_estimate <= upper_max_current
