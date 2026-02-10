import jax.numpy as jnp
from equinox import Module
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Scalar


class ElectrodeKineticsParameters(Module):
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


class ElectrodeKineticsParameters2(Module):
    a: Scalar
    k: Scalar
    e: Scalar
    dB: Scalar

    def __init__(self, alpha: Scalar, kappa: Scalar, epsilon: Scalar, dB: Scalar):
        self.a = logit(alpha)
        self.k = jnp.log(kappa)
        self.e = 5.0 * jnp.arctanh(epsilon / 10.0)
        self.dB = dB

    @property
    def alpha(self):
        return sigmoid(self.a)

    @property
    def kappa(self):
        return jnp.exp(self.k)

    @property
    def epsilon(self):
        return 10.0 * jnp.tanh(self.e / 5.0)


class LinearECIrreversibleParams(Module):
    a: Scalar
    k: Scalar
    km: Scalar
    kp: Scalar
    db: Scalar
    e: Scalar

    def __init__(
        self,
        alpha: Scalar,
        kappa: Scalar,
        kappam: Scalar,
        kappap: Scalar,
        deltab: Scalar,
        epsilon: Scalar,
    ):
        self.a = logit(alpha)
        self.k = jnp.log(kappa)
        self.km = jnp.log(kappam)
        self.kp = jnp.log(kappap)
        self.db = deltab
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

    @property
    def kappam(self):
        return jnp.exp(self.k)

    @property
    def kappap(self):
        return jnp.exp(self.k)

    @property
    def deltab(self):
        return self.db
