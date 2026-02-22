from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.fdm import (
    SecondOrderECirreFDMSolverBackwardImplicit,
    SecondOrderECirreFDMSolverExplicit,
    SecondOrderECirreFDMSolverNewton,
)
from src.params import SecondOrderECirreMechanismFDMParams
from src.voltammetry import CyclicDC

# jax.config.update("jax_enable_x64", True)

voltammetry = CyclicDC()

newton_solver = SecondOrderECirreFDMSolverNewton(voltammetry, h=1e-3, dtheta=5e-2)

explicit_solver = SecondOrderECirreFDMSolverExplicit(voltammetry, h=1e-3, dtheta=5e-2)

backward_solver = SecondOrderECirreFDMSolverBackwardImplicit(
    voltammetry, h=1e-3, dtheta=5e-2
)

kp_range = jnp.power(10, jnp.linspace(4, 8, 5))
km_range = kp_range / 1e6

base_params = SecondOrderECirreMechanismFDMParams(
    alpha=jnp.array(1.0),
    K0=jnp.array(10000.0),
    Kplus=jnp.array(1000000.0),
    Kminus=jnp.array(1.0),
    dB=jnp.array(1.0),
    dY=jnp.array(1.0),
    dZ=jnp.array(1.0),
    E0=jnp.array(0.0),
)

# params = SecondOrderECirreMechanismFDMParams(
#     alpha=jnp.full_like(kp_range, base_params.alpha),
#     K0=jnp.full_like(kp_range, base_params.K0),
#     Kplus=kp_range,
#     Kminus=km_range,
#     dB=jnp.full_like(kp_range, base_params.dB),
#     dY=jnp.full_like(kp_range, base_params.dY),
#     dZ=jnp.full_like(kp_range, base_params.dZ),
#     E0=jnp.full_like(kp_range, base_params.E0),
# )

start = perf_counter()
base_current_exp = explicit_solver.solve(base_params)
base_current_exp.block_until_ready()
end = perf_counter()
print(f"Explicit: {end - start:.4f}")

start = perf_counter()
base_current_back = backward_solver.solve(base_params)
base_current_back.block_until_ready()
end = perf_counter()
print(f"Backward Implicit: {end - start:.4f}")

start = perf_counter()
base_current_new = newton_solver.solve(base_params)
base_current_new.block_until_ready()
end = perf_counter()
print(f"Newton: {end - start:.4f}")
print(f"DType:{base_current_new.dtype}")

# currents = jax.vmap(solver.solve)(params)
#
# for c, kp in zip(currents, kp_range):
#     plt.plot(solver.applied_potentials, c, label=f"{kp:.2e}")

options = {"alpha": 0.5}

plt.plot(
    explicit_solver.applied_potentials,
    base_current_exp,
    linestyle="-.",
    label="Explicit",
    **options,
)

plt.plot(
    backward_solver.applied_potentials,
    base_current_back,
    label="Backward Implicit",
    linestyle="--",
    **options,
)

plt.plot(newton_solver.applied_potentials, base_current_new, label="Newton", **options)

plt.legend()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
