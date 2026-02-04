import jax.numpy as jnp
from equinox import Module
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Scalar


class MacroElectrodeParams(Module):
    a: Scalar
    k: Scalar
    e: Scalar

    def __init__(self, alpha: Scalar, kappa: Scalar, epsilon: Scalar):
        self.a = logit(alpha)
        self.k = jnp.log(kappa)
        self.e = 5.0 * jnp.arctanh(epsilon / 10.0)

    @property
    def alpha(self):
        return sigmoid(self.a)

    @property
    def kappa(self):
        return jnp.exp(self.k)

    @property
    def epsilon(self):
        return 10.0 * jnp.tanh(self.e / 5.0)
