import jax.numpy as jnp
from equinox import Module
from jax import Array
from jaxtyping import ArrayLike, Scalar


class ButlerVolmerParameters(Module):
    alpha: Scalar
    k0: Scalar

    def __init__(self, alpha: float, k0: float):
        self.alpha = jnp.array([alpha])
        self.k0 = jnp.array([k0])

    def __str__(self):
        return f"a={self.alpha},k0={self.k0}"


class ButlerVolmerElectrodeKinetics:
    def alpha(self, X: Array, theta: float, params: ButlerVolmerParameters):
        h = X[1] - X[0]
        inner = h / (X[1:-1] - X[:-2]) ** 2
        return jnp.concatenate(
            [
                jnp.array([0.0]),
                -inner,
                jnp.array([0.0]),
            ]
        )

    def beta(self, X: Array, theta: float, params: ButlerVolmerParameters):
        h = X[1] - X[0]
        inner = h / (X[1:-1] - X[:-2]) ** 2
        b0 = 1 + h * jnp.exp(-params.alpha * theta) * params.k0 * (1 + jnp.exp(theta))
        return jnp.concatenate(
            [
                b0,
                1 + 2 * inner,
                jnp.array([1.0]),
            ],
        )

    def sigma(self, X: Array, theta: float, params: ButlerVolmerParameters):
        h = X[1] - X[0]
        inner = h / (X[1:-1] - X[:-2]) ** 2
        return jnp.concatenate(
            [
                jnp.array([-1.0]),
                -inner,
                jnp.array([0.0]),
            ]
        )

    def delta(
        self,
        c_prev: Array,
        X: Array,
        theta: ArrayLike,
        params: ButlerVolmerParameters,
    ) -> Array:
        h = X[1] - X[0]
        inner = c_prev[1:-1]
        d0 = h * jnp.exp(-params.alpha * theta) * params.k0 * jnp.exp(theta)
        return jnp.concatenate(
            [
                d0,
                inner,
                jnp.array([1.0]),
            ]
        )
