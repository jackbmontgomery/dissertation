from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from equinox import apply_updates, filter, filter_jit, filter_value_and_grad, is_array
from jax import debug, vmap

from src.electrode_kinetics import ButlerVolmerElectrodeKinetics, ButlerVolmerParameters
from src.experiment import CyclicMacroBand1D
from src.fdm import ImplicitFDSolver1D

theta_i = 20.0
theta_v = -20.0
sigma = 100.0

dtheta = 0.02
dx = 2e-4
dt = dtheta / sigma

experiment = CyclicMacroBand1D(theta_i, theta_v, sigma)
electrode_kinetics = ButlerVolmerElectrodeKinetics()
solver = ImplicitFDSolver1D()

X = jnp.linspace(
    experiment.x_min, experiment.x_max, int((experiment.x_max - experiment.x_min) / dx)
)

T = jnp.linspace(
    experiment.t_min, experiment.t_max, int((experiment.t_max - experiment.t_min) / dx)
)


c_init = jnp.ones_like(X)

potentials = vmap(experiment.potential)(T)


def simulate(params):
    _, current = solver.solve(c_init, X, potentials, electrode_kinetics, dx, params)
    return current


true_current = simulate(ButlerVolmerParameters(0.4, 10.0))

opt = optax.adam(1e-1)


def loss_fn(params, true_current):
    pred_current = simulate(params)
    return jnp.mean(jnp.square(pred_current - true_current))


@filter_jit
def make_step(params, opt_state, true_current):
    loss, grads = filter_value_and_grad(loss_fn)(params, true_current)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = apply_updates(params, updates)
    return loss, params, opt_state


params = ButlerVolmerParameters(0.3, 6.0)
opt_state = opt.init(filter(params, is_array))

ini_current = simulate(params)

start = perf_counter()
for i in range(10):
    loss, params, opt_state = make_step(params, opt_state, true_current)
    print(i, loss)

end = perf_counter()

print("Avg:", (end - start) / 10)

debug.print("{x}", x=params)
pred_current = simulate(params)
plt.plot(potentials, true_current, label="True")
plt.plot(potentials, ini_current, label="Init")
plt.plot(potentials, pred_current, label="Pred")
plt.legend()
plt.show()
