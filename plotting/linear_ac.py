import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jaxtyping import Scalar

from src.experiment import LinearSweepACMacroBand, LinearSweepDCMacroBand
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    uniform_discretise,
)
from src.pde_parameters import ButlerVolmerPhysicalParameters
from src.simulate import create_fdm_current_simulator


def main(
    alpha: Scalar,
    kappa0: Scalar,
    amplitude: float = 0.25,
    num_oscillations: int = 32,
):
    experiment = LinearSweepACMacroBand(
        amplitude=amplitude, num_oscillations=num_oscillations
    )
    T, X = uniform_discretise(experiment)

    print(f"T:{T.shape},X:{X.shape}")

    potentials = vmap(experiment.potential)(T)

    fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, T)

    c_init = jnp.ones_like(X)

    simulate_current = create_fdm_current_simulator(
        c_init, potentials, fdm_discretisation, X
    )

    params = ButlerVolmerPhysicalParameters(alpha=alpha, kappa0=kappa0)

    current = simulate_current(params)
    linear_potentials = vmap(LinearSweepDCMacroBand().potential)(T)

    plt.plot(linear_potentials, current)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()


if __name__ == "__main__":
    main(alpha=jnp.array(0.8), kappa0=jnp.array(100.0))
