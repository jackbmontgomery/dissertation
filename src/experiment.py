from math import sqrt

from equinox import Module, field
from jax.lax import cond


class LinearSweepMacroBand(Module):
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
