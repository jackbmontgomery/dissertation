from abc import abstractmethod
from math import sqrt

import jax.numpy as jnp
from equinox import AbstractVar, Module, field
from jax.lax import cond


class AbstractExperiment(Module):
    theta_i: AbstractVar[float]
    theta_v: AbstractVar[float]
    sigma: AbstractVar[float]
    t_min: AbstractVar[float]
    t_max: AbstractVar[float]
    x_min: AbstractVar[float]
    x_max: AbstractVar[float]

    @abstractmethod
    def potential(self, t: float) -> float:
        raise NotImplementedError


class LinearSweepACMacroBand(AbstractExperiment):
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
        sigma: float = 1000.0,
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


class LinearSweepDCMacroBand(AbstractExperiment):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    x_min: float = field(static=True)
    x_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 1000.0
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


class CyclicMacroBand1D(AbstractExperiment):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    x_min: float = field(static=True)
    x_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 1000.0
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
