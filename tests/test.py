import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import FirstOrderECirreMechanismFDMSolver
from src.params import FirstOrderECirreMechanismFDMParams
from src.voltammetry import CyclicDC

voltammetry = CyclicDC()
fdm_solver = FirstOrderECirreMechanismFDMSolver(voltammetry)

base_params = FirstOrderECirreMechanismFDMParams(
    alpha=jnp.array(1.0),
    K0=jnp.array(100000.0),
    K1=jnp.array(1.0),
    E0=jnp.array(0.0),
    dB=jnp.array(1.0),
)

K1_range = jnp.array([1.0, 5.0, 10.0, 50.0, 100.0])

params = FirstOrderECirreMechanismFDMParams(
    alpha=jnp.full_like(K1_range, base_params.alpha),
    K0=jnp.full_like(K1_range, base_params.K0),
    K1=K1_range,
    E0=jnp.full_like(K1_range, base_params.E0),
    dB=jnp.full_like(K1_range, base_params.dB),
)

currents = vmap(fdm_solver.solve)(params)

for c, K1 in zip(currents, K1_range):
    plt.plot(fdm_solver.applied_potentials, c, label=K1)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()
plt.show()
