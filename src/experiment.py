from math import sqrt

import jax.numpy as jnp
from equinox import Module, field
from jax.lax import cond


class LinearSweepACMacroBand(Module):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    x_min: float = field(static=True)
    x_max: float = field(static=True)
    e0: float = field(static=True)
    omega: float = field(static=True)

    def __init__(
        self,
        theta_i: float = 10.0,
        theta_v: float = -10.0,
        sigma: float = 100.0,
        amplitude: float = 0.25,
        num_oscillations: int = 32,
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = abs(theta_v - theta_i) / sigma
        self.x_min = 0.0
        self.x_max = 6.0 * sqrt(self.t_max)

        self.e0 = amplitude
        self.omega = 2 * num_oscillations * jnp.pi / self.t_max

    def potential(self, t: float) -> float:
        return self.theta_i - self.sigma * t - self.e0 * jnp.sin(self.omega * t)


class LinearSweepDCMacroBand(Module):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    x_min: float = field(static=True)
    x_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 100.0
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = abs(theta_v - theta_i) / sigma
        self.x_min = 0.0
        self.x_max = 6.0 * sqrt(self.t_max)

    def potential(self, t: float) -> float:
        return self.theta_i - self.sigma * t


class CyclicMacroBand1D(Module):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    x_min: float = field(static=True)
    x_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 100.0
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = 2.0 * abs(theta_v - theta_i) / sigma
        self.x_min = 0.0
        self.x_max = 6.0 * sqrt(self.t_max)

    def potential(self, t: float) -> float:
        theta = cond(
            t < self.t_max / 2.0,
            lambda t: self.theta_i - self.sigma * t,
            lambda t: self.theta_v + self.sigma * (t - self.t_max / 2.0),
            t,
        )
        return theta
