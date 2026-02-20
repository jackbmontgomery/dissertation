import jax.numpy as jnp
from equinox import Module
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Scalar


class EMechanismFDMParams(Module):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _E0_inv: Scalar
    _dB_inv: Scalar

    def __init__(self, alpha: Scalar, K0: Scalar, E0: Scalar, dB: Scalar):
        self._alpha_inv = 3.0 * logit(alpha)
        self._K0_inv = jnp.log10(K0)
        self._E0_inv = 5.0 * jnp.arctanh(E0 / 20.0)
        self._dB_inv = jnp.log2(dB)

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv / 3.0)

    @property
    def K0(self):
        return jnp.power(10, self._K0_inv)

    @property
    def E0(self):
        return 20.0 * jnp.tanh(self._E0_inv / 5.0)

    @property
    def dB(self):
        return jnp.power(2, self._dB_inv)


class ECirreMechanismFDMParams(Module):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _Kplus_inv: Scalar
    _Kminus_inv: Scalar
    _E0_inv: Scalar
    _dB_inv: Scalar

    def __init__(
        self,
        alpha: Scalar,
        K0: Scalar,
        Kminus: Scalar,
        Kplus: Scalar,
        dB: Scalar,
        E0: Scalar,
    ):
        self._alpha_inv = 3.0 * logit(alpha)
        self._K0_inv = jnp.log10(K0)
        self._Kplus_inv = jnp.log(Kminus)
        self._Kminus_inv = jnp.log(Kplus)
        self._E0_inv = 5.0 * jnp.arctanh(E0 / 20.0)
        self._dB_inv = jnp.log2(dB)

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv / 3.0)

    @property
    def K0(self):
        return jnp.power(10, self._K0_inv)

    @property
    def E0(self):
        return 20.0 * jnp.tanh(self._E0_inv / 5.0)

    @property
    def dB(self):
        return jnp.power(2, self._dB_inv)

    @property
    def Kminus(self):
        return jnp.exp(self._Kplus_inv)

    @property
    def Kplus(self):
        return jnp.exp(self._Kminus_inv)


class SecondOrderECirreMechanismFDMParams(Module):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _Kplus_inv: Scalar
    _Kminus_inv: Scalar
    _E0_inv: Scalar
    _dB_inv: Scalar
    _dY_inv: Scalar
    _dZ_inv: Scalar

    def __init__(
        self,
        alpha: Scalar,
        K0: Scalar,
        Kminus: Scalar,
        Kplus: Scalar,
        dB: Scalar,
        dY: Scalar,
        dZ: Scalar,
        E0: Scalar,
    ):
        self._alpha_inv = 3.0 * logit(alpha)
        self._K0_inv = jnp.log10(K0)
        self._Kplus_inv = jnp.log(Kminus)
        self._Kminus_inv = jnp.log(Kplus)
        self._E0_inv = 5.0 * jnp.arctanh(E0 / 20.0)
        self._dB_inv = jnp.log2(dB)
        self._dY_inv = jnp.log2(dY)
        self._dZ_inv = jnp.log2(dZ)

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv / 3.0)

    @property
    def K0(self):
        return jnp.power(10, self._K0_inv)

    @property
    def E0(self):
        return 20.0 * jnp.tanh(self._E0_inv / 5.0)

    @property
    def dB(self):
        return jnp.power(2, self._dB_inv)

    @property
    def dY(self):
        return jnp.power(2, self._dB_inv)

    @property
    def dZ(self):
        return jnp.power(2, self._dB_inv)

    @property
    def Kminus(self):
        return jnp.exp(self._Kplus_inv)

    @property
    def Kplus(self):
        return jnp.exp(self._Kminus_inv)
