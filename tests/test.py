from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from equinox import apply_updates, filter, filter_jit, filter_value_and_grad, is_array
from jax import debug, vmap
from jaxtyping import Scalar

from src.experiment import CyclicMacroBand1D
from src.fdm import ButlerVolmerFDMDiscretisation1D, fdm_implicit_solve
from src.pde_parameters import ButlerVolmerParameters

theta_i = 20.0
theta_v = -20.0
sigma = 100.0

dtheta = 0.02
dx = 2e-4
dt = dtheta / sigma

experiment = CyclicMacroBand1D(theta_i, theta_v, sigma)

X = jnp.linspace(
    experiment.x_min, experiment.x_max, int((experiment.x_max - experiment.x_min) / dx)
)

T = jnp.linspace(
    experiment.t_min, experiment.t_max, int((experiment.t_max - experiment.t_min) / dt)
)

potentials = vmap(experiment.potential)(T)

pde_discretisation = ButlerVolmerFDMDiscretisation1D(X)
c_init = jnp.ones_like(X)


def simulate(
    params: ButlerVolmerParameters,
    c_init=c_init,
    pde_discretisation=pde_discretisation,
    dx=dx,
    potentials=potentials,
):
    _, current = fdm_implicit_solve(c_init, pde_discretisation, params, dx, potentials)
    return current


def loss_fn(params: ButlerVolmerParameters, target_current: Scalar):
    pred_current = simulate(params)
    return jnp.mean(jnp.square(pred_current - target_current))


opt = optax.adam(1e-2)


@filter_jit
def make_step(params, opt_state, target):
    loss, grads = filter_value_and_grad(loss_fn)(params, target)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = apply_updates(params, updates)
    return loss, params, opt_state


target_current = simulate(ButlerVolmerParameters(alpha=0.6, k0=10.0))


params = ButlerVolmerParameters(alpha=0.8, k0=100.0)
opt_state = opt.init(filter(params, is_array))

start = perf_counter()
for i in range(10):
    loss, params, opt_state = make_step(params, opt_state, target_current)
    print(i, loss)

end = perf_counter()

print("Avg:", (end - start) / 10)
debug.print("{x}", x=params)
