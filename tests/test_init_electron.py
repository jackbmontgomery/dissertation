import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from chex import PRNGKey
from jax import vmap
from matplotlib.lines import lineStyles
from PIL.Image import init

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.sampling import NUM_CPUS, AbstractSamplingAlgorithm
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicAC, CyclicDC, VoltammetryType


def create_init_params(key: PRNGKey, num_chains: int):
    k1, k2, k3 = jr.split(key, 3)

    alpha_vals = jnp.linspace(0.5, 0.7, num_chains)
    K0_vals = jnp.linspace(5.0, 20.0, num_chains)
    Ef_vals = jnp.linspace(0.0, 1.0, num_chains)

    alpha = jr.permutation(k1, alpha_vals)
    K0 = jr.permutation(k2, K0_vals)
    Ef = jr.permutation(k3, Ef_vals)

    return ElectronReactionParams(
        alpha=alpha,
        K0=K0,
        Ef=Ef,
    )


true_parameters = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.5),
)


def run(
    noise: float,
    sigma: int,
    voltammetry_type: VoltammetryType,
    seed: int = 42,
):
    key = jr.key(seed)

    if voltammetry_type == "DC":
        voltammetry = CyclicDC(sigma=sigma)
    else:
        voltammetry = CyclicAC(sigma=sigma)

    param_key, sampling_key, key = jr.split(key, 3)

    fdm_solver = ElectronReactionFDSolver(voltammetry)

    _, base_current = fdm_solver.solve(true_parameters)

    samples = generate_noisy_samples(
        10,
        base_current,
        noise,
        key=key,
    )

    def logdensity_fn(params: ElectronReactionParams, samples=samples):
        _, current = fdm_solver.solve(params)
        return -jnp.sum((samples - current) ** 2)

    init_params = create_init_params(param_key, NUM_CPUS)
    # print(init_params.alpha[2], init_params.K0[2], init_params.Ef[2])
    _, init_currents = vmap(fdm_solver.solve)(init_params)
    for i, current in enumerate(init_currents):
        plt.plot(fdm_solver.applied_potentials, current, label=i)

    _, true_current = fdm_solver.solve(true_parameters)
    plt.plot(
        fdm_solver.applied_potentials,
        true_current,
        color="black",
        linestyle="--",
        label="True",
    )
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


run(0.25, 1000, "DC")
