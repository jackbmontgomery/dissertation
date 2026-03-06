from abc import abstractmethod
from typing import Literal

import jax.numpy as jnp
from equinox import AbstractVar, Module, field
from jax.lax import cond
from jaxtyping import Scalar

VoltammetryType = Literal["AC", "DC"]


class AbstractVoltammetryTechnique(Module):
    theta_i: AbstractVar[float]
    theta_v: AbstractVar[float]
    sigma: AbstractVar[float]
    t_min: AbstractVar[float]
    t_max: AbstractVar[float]

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def applied_potential(self, t: Scalar) -> Scalar:
        raise NotImplementedError


class LinearSweepAC(AbstractVoltammetryTechnique):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
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

        self.e0 = amplitude
        self.omega = 2 * num_oscillations * jnp.pi / self.t_max

    def __str__(self):
        return "LinearSweepAC"

    def applied_potential(self, t: Scalar) -> Scalar:
        return self.theta_i - self.sigma * t - self.e0 * jnp.sin(self.omega * t)


class LinearSweepDC(AbstractVoltammetryTechnique):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 1000.0
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = abs(theta_v - theta_i) / sigma

    def __str__(self):
        return "LinearSweepDC"

    def applied_potential(self, t: Scalar) -> Scalar:
        return self.theta_i - self.sigma * t


class CyclicDC(AbstractVoltammetryTechnique):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)

    def __init__(
        self, theta_i: float = 10.0, theta_v: float = -10.0, sigma: float = 1000.0
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = 2.0 * abs(theta_v - theta_i) / sigma

    def __str__(self):
        return "CyclicDC"

    def applied_potential(self, t: Scalar) -> Scalar:
        theta = cond(
            t < self.t_max / 2.0,
            lambda t: self.theta_i - self.sigma * t,
            lambda t: self.theta_v + self.sigma * (t - self.t_max / 2.0),
            t,
        )
        return theta


class CyclicAC(AbstractVoltammetryTechnique):
    theta_i: float = field(static=True)
    theta_v: float = field(static=True)
    sigma: float = field(static=True)
    t_min: float = field(static=True)
    t_max: float = field(static=True)
    e0: float = field(static=True)
    omega: float = field(static=True)

    def __init__(
        self,
        theta_i: float = 10.0,
        theta_v: float = -10.0,
        sigma: float = 1000.0,
        amplitude: float = 0.5,
        num_oscillations: int = 64,
    ):
        self.theta_i = theta_i
        self.theta_v = theta_v
        self.sigma = sigma

        self.t_min = 0.0
        self.t_max = 2.0 * abs(theta_v - theta_i) / sigma

        self.e0 = amplitude
        self.omega = 2 * num_oscillations * jnp.pi / self.t_max

    def __str__(self):
        return "CyclicAC"

    def applied_potential(self, t: Scalar) -> Scalar:
        theta = cond(
            t < self.t_max / 2.0,
            lambda t: self.theta_i - self.sigma * t - self.e0 * jnp.sin(self.omega * t),
            lambda t: self.theta_v
            + self.sigma * (t - self.t_max / 2.0)
            - self.e0 * jnp.sin(self.omega * t),
            t,
        )
        return theta
