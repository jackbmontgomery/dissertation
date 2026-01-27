import jax.numpy as jnp
from equinox import Module
from jaxtyping import Scalar


class ButlerVolmerParameters(Module):
    alpha: Scalar
    k0: Scalar

    def __init__(self, alpha: float, k0: float):
        self.alpha = jnp.array([alpha])
        self.k0 = jnp.array([k0])

    def __str__(self):
        return f"a={self.alpha},k0={self.k0}"
