import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from src.fdm import (
    EMechanismFDMSolver,
    HeterogeneousECirreFDMSolver,
    HeterogeneousECirreTestFDSolver,
)
from src.params import EMechanismFDMParams, HeterogenousECirreMechanismFDMParams
from src.voltammetry import CyclicDC

voltammetry = CyclicDC()
hetro_fdm_solver = HeterogeneousECirreFDMSolver(voltammetry)
# hetro_fdm_solver = HeterogeneousECirreTestFDSolver(voltammetry)
e_fdm_solver = EMechanismFDMSolver(voltammetry)

hetro_params = HeterogenousECirreMechanismFDMParams(
    alpha1=jnp.array(1.0),
    K1_0=jnp.array(100000.0),
    E1_f=jnp.array(0.0),
    alpha2=jnp.array(0.0),
    K2_0=jnp.array(0.0),
    E2_f=jnp.array(0.0),
    dB=jnp.array(1.0),
    dC=jnp.array(1.0),
    dD=jnp.array(1.0),
    K_het=jnp.array(0.0),
)

e_params = EMechanismFDMParams(
    alpha=jnp.array(1.0),
    K0=jnp.array(100000.0),
    E0=jnp.array(0.0),
    dB=jnp.array(1.0),
)

hetro_current = hetro_fdm_solver.solve(hetro_params)
# hetro_current = hetro_fdm_solver.solve(e_params)
# e_current = e_fdm_solver.solve(e_params)
plt.plot(hetro_fdm_solver.applied_potentials, hetro_current)
# plt.plot(e_fdm_solver.applied_potentials, e_current)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

# K1_range = jnp.array([1.0, 5.0, 10.0, 50.0, 100.0])
#
# params = FirstOrderECirreMechanismFDMParams(
#     alpha=jnp.full_like(K1_range, base_params.alpha),
#     K0=jnp.full_like(K1_range, base_params.K0),
#     K1=K1_range,
#     E0=jnp.full_like(K1_range, base_params.E0),
#     dB=jnp.full_like(K1_range, base_params.dB),
# )
#
# currents = vmap(fdm_solver.solve)(params)
#
# for c, K1 in zip(currents, K1_range):
#     plt.plot(fdm_solver.applied_potentials, c, label=K1)
#
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.legend()
# plt.show()
