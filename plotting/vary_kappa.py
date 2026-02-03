import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from src.experiment import LinearSweepDCMacroBand
from src.fdm_discretisation import (
    ButlerVolmerFDMDiscretisation1D,
    uniform_discretise,
)
from src.pde_parameters import ButlerVolmerPhysicalParameters
from src.simulate import create_fdm_current_simulator


def main():
    experiment = LinearSweepDCMacroBand()
    T, X = uniform_discretise(experiment)

    print(f"T:{T.shape},X:{X.shape}")

    potentials = vmap(experiment.potential)(T)

    fdm_discretisation = ButlerVolmerFDMDiscretisation1D(X, T)

    c_init = jnp.ones_like(X)

    simulate_current = create_fdm_current_simulator(
        c_init, potentials, fdm_discretisation, X
    )

    kappas = jnp.linspace(1, 5, 5)
    for k in kappas:
        params = ButlerVolmerPhysicalParameters(
            alpha=jnp.array(1.0), kappa0=jnp.power(10.0, k)
        )
        current = simulate_current(params)

        plt.plot(potentials, current, label=f"{jnp.power(10.0, k):.1f}")

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
