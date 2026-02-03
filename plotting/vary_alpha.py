import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jaxtyping import Scalar

from src.experiment import LinearSweepDCMacroBand
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    uniform_discretise,
)
from src.pde_parameters import ButlerVolmerPhysicalParameters
from src.simulate import create_fdm_current_simulator

"""
NOTE:
It is very important that sigma is large since that distinguishes how large the peak of the current it.
If it is too small then alpha is very hard to determine that parameter
"""


def main(kappa0: Scalar):
    experiment = LinearSweepDCMacroBand()
    T, X = uniform_discretise(experiment)

    print(f"T:{T.shape},X:{X.shape}")

    potentials = vmap(experiment.potential)(T)

    fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, T)

    c_init = jnp.ones_like(X)

    simulate_current = create_fdm_current_simulator(
        c_init, potentials, fdm_discretisation, X
    )

    alpahs = jnp.linspace(0.3, 0.7, 5)
    for a in alpahs:
        params = ButlerVolmerPhysicalParameters(alpha=jnp.array(a), kappa0=kappa0)
        current = simulate_current(params)

        plt.plot(potentials, current, label=f"{a:.1f}")

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(kappa0=jnp.array(100.0))
