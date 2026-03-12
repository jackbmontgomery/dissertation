import cma
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.utils import generate_noisy_samples
from src.voltammetry import CyclicDC

key = jr.key(0)
cyclic_dc = CyclicDC()
fd_solver = ElectronReactionFDSolver(cyclic_dc)

true_params = ElectronReactionParams(
    alpha=jnp.array(0.6),
    K0=jnp.array(10.0),
    Ef=jnp.array(0.0),
)
init_params = ElectronReactionParams(
    alpha=jnp.array(0.5),
    K0=jnp.array(15.0),
    Ef=jnp.array(0.5),
)

_, base_current = fd_solver.solve(true_params)
experimental_samples = generate_noisy_samples(10, base_current, 0.25, key=key)


def log_density(params: ElectronReactionParams, samples=experimental_samples):
    _, current = fd_solver.solve(params)
    return -jnp.sum((samples - current) ** 2)


PARAM_NAMES = ["alpha", "K0", "Ef"]


def params_to_vec(params: ElectronReactionParams) -> np.ndarray:
    """Flatten ElectronReactionParams to a plain numpy vector."""
    return np.array(
        [
            float(params.alpha),
            float(params.K0),
            float(params.Ef),
        ]
    )


def vec_to_params(x: np.ndarray) -> ElectronReactionParams:
    """Reconstruct ElectronReactionParams from a numpy vector."""
    return ElectronReactionParams(
        alpha=jnp.array(x[0]),
        K0=jnp.array(x[1]),
        Ef=jnp.array(x[2]),
    )


def objective(x: np.ndarray) -> float:
    """CMA-ES minimises, so negate the log-density."""
    params = vec_to_params(x)
    return -float(log_density(params))


# --- Run CMA-ES ---

x0 = params_to_vec(init_params)
sigma0 = 10.0

bounds = [
    [1e-6, 1e-6, -10.0],  # lower bounds
    [1.0, 100.0, 10.0],
]  # upper bounds

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": bounds,
        "tolx": 1e-4,
        "tolfun": 1e-6,
        "maxfevals": 50,
        "verbose": 1,
    },
)

history = []

while not es.stop():
    solutions = es.ask()
    fitnesses = [objective(x) for x in solutions]
    es.tell(solutions, fitnesses)
    history.append(es.result.xbest)
    es.disp()

# --- Results ---
result = es.result
best_params = vec_to_params(result.xbest)
np.savez_compressed("./data/temp_cma.npz", params=history)

print("\n=== CMA-ES Result ===")
print(f"  alpha : {float(best_params.alpha):.6f}  (true: 0.6)")
print(f"  K0    : {float(best_params.K0):.6f}  (true: 10.0)")
print(f"  Ef    : {float(best_params.Ef):.6f}  (true: 0.0)")
print(f"  log-density at optimum : {-result.fbest:.4f}")
print(f"  total likelihood evals : {result.evaluations}")
