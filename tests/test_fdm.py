import jax.numpy as jnp
from equinox import filter_value_and_grad
from jax import grad, vmap
from jax.lax import linalg
from jaxtyping import Array

from src.common import compute_current
from src.electrode_kinetics import ButlerVolmerElectrodeKinetics, ButlerVolmerParameters
from src.experiment import CyclicMacroBand1D
from src.fdm import ImplicitFDSolver1D


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return linalg.tridiagonal_solve(dl, d, du, b[:, None]).flatten()


def solve(
    c_init: Array,
    electrode_kinetics: ButlerVolmerElectrodeKinetics,
    X: Array,
    potentials: Array,
    dx: float,
    params: ButlerVolmerParameters,
) -> Array:
    def fdm_stepper(c_prev, theta):
        dl = electrode_kinetics.alpha(X, theta, params)
        d = electrode_kinetics.beta(X, theta, params)
        du = electrode_kinetics.sigma(X, theta, params)
        rhs = electrode_kinetics.delta(c_prev, X, theta, params)

        ck = tridiagonal_solve(dl, d, du, rhs)

        current = compute_current(ck, dx)

        return ck, current

    _, current = fdm_stepper(c_init, potentials)

    return current


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

potentials = experiment.potential(0.0)

target_params = ButlerVolmerParameters(0.6, 10.0)


def simulate(params):
    current = solve(c_init, electrode_kinetics, X, potentials, dx, params)
    return current


def loss(params, true_current):
    pred_current = simulate(params)
    return jnp.mean(jnp.square(pred_current - true_current))


value_grad = filter_value_and_grad(loss)

target_current = simulate(target_params)
params = ButlerVolmerParameters(0.8, 15.0)

print(value_grad(params, target_current))
