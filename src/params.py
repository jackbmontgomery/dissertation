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


class FirstOrderECirreMechanismFDMParams(Module):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _K1_inv: Scalar
    _E0_inv: Scalar
    _dB_inv: Scalar

    def __init__(self, alpha: Scalar, K0: Scalar, K1: Scalar, E0: Scalar, dB: Scalar):
        self._alpha_inv = 3.0 * logit(alpha)
        self._K0_inv = jnp.log10(K0)
        self._K1_inv = jnp.log10(K1)
        self._E0_inv = 5.0 * jnp.arctanh(E0 / 20.0)
        self._dB_inv = jnp.log2(dB)

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv / 3.0)

    @property
    def K0(self):
        return jnp.power(10, self._K0_inv)

    @property
    def K1(self):
        return jnp.power(10, self._K1_inv)

    @property
    def E0(self):
        return 20.0 * jnp.tanh(self._E0_inv / 5.0)

    @property
    def dB(self):
        return jnp.power(2, self._dB_inv)


class HeterogenousECirreMechanismFDMParams(Module):
    _alpha1_inv: Scalar
    _K1_0_inv: Scalar
    _E1_f_inv: Scalar
    _alpha2_inv: Scalar
    _K2_0_inv: Scalar
    _E2_f_inv: Scalar
    _dB_inv: Scalar
    _dC_inv: Scalar
    _dD_inv: Scalar
    _K_het_inv: Scalar

    def __init__(
        self,
        alpha1: Scalar,
        K1_0: Scalar,
        E1_f: Scalar,
        alpha2: Scalar,
        K2_0: Scalar,
        E2_f: Scalar,
        dB: Scalar,
        dC: Scalar,
        dD: Scalar,
        K_het: Scalar,
    ):
        self._alpha1_inv = 3.0 * logit(alpha1)
        self._K1_0_inv = jnp.log10(K1_0)
        self._E1_f_inv = 5.0 * jnp.arctanh(E1_f / 20.0)

        self._alpha2_inv = 3.0 * logit(alpha2)
        self._K2_0_inv = jnp.log10(K2_0)
        self._E2_f_inv = 5.0 * jnp.arctanh(E2_f / 20.0)

        self._dB_inv = jnp.log2(dB)
        self._dC_inv = jnp.log2(dC)
        self._dD_inv = jnp.log2(dD)

        self._K_het_inv = jnp.log10(K_het)

    @property
    def alpha1(self):
        return sigmoid(self._alpha1_inv / 3.0)

    @property
    def alpha2(self):
        return sigmoid(self._alpha2_inv / 3.0)

    @property
    def K1_0(self):
        return jnp.power(10, self._K1_0_inv)

    @property
    def K2_0(self):
        return jnp.power(10, self._K2_0_inv)

    @property
    def E1_f(self):
        return 20.0 * jnp.tanh(self._E1_f_inv / 5.0)

    @property
    def E2_f(self):
        return 20.0 * jnp.tanh(self._E2_f_inv / 5.0)

    @property
    def dB(self):
        return jnp.power(2, self._dB_inv)

    @property
    def dC(self):
        return jnp.power(2, self._dC_inv)

    @property
    def dD(self):
        return jnp.power(2, self._dD_inv)

    @property
    def K_het(self):
        return jnp.power(10, self._K_het_inv)


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
        return jnp.power(2, self._dY_inv)

    @property
    def dZ(self):
        return jnp.power(2, self._dZ_inv)

    @property
    def Kminus(self):
        return jnp.exp(self._Kplus_inv)

    @property
    def Kplus(self):
        return jnp.exp(self._Kminus_inv)
