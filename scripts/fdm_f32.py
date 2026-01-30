import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap

from src.experiment import LinearSweepMacroBand
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    discretise_experiment,
)
from src.pde_parameters import ButlerVolmerPhysicalParameters
from src.simulate import create_fdm_current_simulator

key = jr.key(0)

experiment = LinearSweepMacroBand()
dx = 1e-2
T, X = discretise_experiment(experiment, dx=dx)

print(f"T:{X.shape},X:{T.shape}")

potentials = vmap(experiment.potential)(T)

fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X)
c_init = jnp.ones_like(X)

simulate_current = create_fdm_current_simulator(
    c_init, potentials, fdm_discretisation, dx
)
params_1 = ButlerVolmerPhysicalParameters(alpha=jnp.array(0.75), kappa0=jnp.array(1.0))
params_2 = ButlerVolmerPhysicalParameters(
    alpha=jnp.array(0.75), kappa0=jnp.array(100.0)
)

current_1 = simulate_current(params_1)
current_2 = simulate_current(params_2)
plt.plot(potentials, current_1)
plt.plot(potentials, current_2)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()
