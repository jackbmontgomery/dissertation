import inspect

import jax.numpy as jnp
from equinox import Module
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Scalar

Params = Module


def param_property_names(params: Params) -> list[str]:
    return [
        name
        for name, val in inspect.getmembers(
            type(params), lambda v: isinstance(v, property)
        )
        if name != "thetaf_ads"
    ]


class ElectronReactionParams(Params):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _thetaf_inv: Scalar

    def __init__(self, alpha: Scalar, K0: Scalar, thetaf: Scalar):
        self._alpha_inv = logit(alpha)
        self._K0_inv = jnp.log(K0)
        self._thetaf_inv = thetaf

    @classmethod
    def from_transformed(cls, alpha_inv: Scalar, K0_inv: Scalar, thetaf_inv: Scalar):
        obj = object.__new__(cls)
        object.__setattr__(obj, "_alpha_inv", alpha_inv)
        object.__setattr__(obj, "_K0_inv", K0_inv)
        object.__setattr__(obj, "_thetaf_inv", thetaf_inv)
        return obj

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv)

    @property
    def K0(self):
        return jnp.exp(self._K0_inv)

    @property
    def thetaf(self):
        return self._thetaf_inv


class HeterogenousReactionParams(Params):
    _alpha_1_inv: Scalar
    _K0_1_inv: Scalar
    _thetaf_1_inv: Scalar
    _alpha_2_inv: Scalar
    _K0_2_inv: Scalar
    _thetaf_2_inv: Scalar
    _K_het_inv: Scalar

    def __init__(
        self,
        alpha_1: Scalar,
        K0_1: Scalar,
        thetaf_1: Scalar,
        alpha_2: Scalar,
        K0_2: Scalar,
        thetaf_2: Scalar,
        K_het: Scalar,
    ):
        self._alpha_1_inv = logit(alpha_1)
        self._K0_1_inv = jnp.log(K0_1)
        self._thetaf_1_inv = thetaf_1

        self._alpha_2_inv = logit(alpha_2)
        self._K0_2_inv = jnp.log(K0_2)
        self._thetaf_2_inv = thetaf_2

        self._K_het_inv = jnp.log(K_het)

    @classmethod
    def from_transformed(
        cls,
        alpha_1_inv: Scalar,
        K0_1_inv: Scalar,
        thetaf_1_inv: Scalar,
        alpha_2_inv: Scalar,
        K0_2_inv: Scalar,
        thetaf_2_inv: Scalar,
        K_het_inv: Scalar,
    ):
        obj = object.__new__(cls)

        object.__setattr__(obj, "_alpha_1_inv", alpha_1_inv)
        object.__setattr__(obj, "_K0_1_inv", K0_1_inv)
        object.__setattr__(obj, "_thetaf_1_inv", thetaf_1_inv)

        object.__setattr__(obj, "_alpha_2_inv", alpha_2_inv)
        object.__setattr__(obj, "_K0_2_inv", K0_2_inv)
        object.__setattr__(obj, "_thetaf_2_inv", thetaf_2_inv)

        object.__setattr__(obj, "_K_het_inv", K_het_inv)

        return obj

    @property
    def alpha_1(self):
        return sigmoid(self._alpha_1_inv)

    @property
    def alpha_2(self):
        return sigmoid(self._alpha_2_inv)

    @property
    def K0_1(self):
        return jnp.exp(self._K0_1_inv)

    @property
    def K0_2(self):
        return jnp.exp(self._K0_2_inv)

    @property
    def thetaf_1(self):
        return self._thetaf_1_inv

    @property
    def thetaf_2(self):
        return self._thetaf_2_inv

    @property
    def K_het(self):
        return jnp.exp(self._K_het_inv)


class AdsorptionReactionParams(Params):
    _alpha_sol_inv: Scalar
    _K0_sol_inv: Scalar
    _thetaf_sol_inv: Scalar
    _alpha_ads_inv: Scalar
    _K0_ads_inv: Scalar
    _K_A_ads_inv: Scalar
    _K_A_des_inv: Scalar
    _K_B_ads_inv: Scalar
    _K_B_des_inv: Scalar

    def __init__(
        self,
        alpha_sol: Scalar,
        K0_sol: Scalar,
        thetaf_sol: Scalar,
        alpha_ads: Scalar,
        K0_ads: Scalar,
        K_A_ads: Scalar,
        K_A_des: Scalar,
        K_B_ads: Scalar,
        K_B_des: Scalar,
    ):
        self._alpha_sol_inv = logit(alpha_sol)
        self._K0_sol_inv = jnp.log(K0_sol)
        self._thetaf_sol_inv = thetaf_sol

        self._alpha_ads_inv = logit(alpha_ads)
        self._K0_ads_inv = jnp.log(K0_ads)

        self._K_A_ads_inv = jnp.log(K_A_ads)
        self._K_A_des_inv = jnp.log(K_A_des)
        self._K_B_ads_inv = jnp.log(K_B_ads)
        self._K_B_des_inv = jnp.log(K_B_des)

    @classmethod
    def from_transformed(
        cls,
        alpha_sol_inv: Scalar,
        K0_sol_inv: Scalar,
        thetaf_sol_inv: Scalar,
        alpha_ads_inv: Scalar,
        K0_ads_inv: Scalar,
        K_A_ads_inv: Scalar,
        K_A_des_inv: Scalar,
        K_B_ads_inv: Scalar,
        K_B_des_inv: Scalar,
    ):
        obj = object.__new__(cls)

        object.__setattr__(obj, "_alpha_sol_inv", alpha_sol_inv)
        object.__setattr__(obj, "_K0_sol_inv", K0_sol_inv)
        object.__setattr__(obj, "_thetaf_sol_inv", thetaf_sol_inv)

        object.__setattr__(obj, "_alpha_ads_inv", alpha_ads_inv)
        object.__setattr__(obj, "_K0_ads_inv", K0_ads_inv)

        object.__setattr__(obj, "_K_A_ads_inv", K_A_ads_inv)
        object.__setattr__(obj, "_K_A_des_inv", K_A_des_inv)

        object.__setattr__(obj, "_K_B_ads_inv", K_B_ads_inv)
        object.__setattr__(obj, "_K_B_des_inv", K_B_des_inv)

        return obj

    @property
    def alpha_sol(self):
        return sigmoid(self._alpha_sol_inv)

    @property
    def K0_sol(self):
        return jnp.exp(self._K0_sol_inv)

    @property
    def thetaf_sol(self):
        return self._thetaf_sol_inv

    @property
    def alpha_ads(self):
        return sigmoid(self._alpha_ads_inv)

    @property
    def K0_ads(self):
        return jnp.exp(self._K0_ads_inv)

    @property
    def thetaf_ads(self):
        return self.thetaf_sol - jnp.log(
            (self.K_A_ads / self.K_A_des) / (self.K_B_ads / self.K_B_des)
        )

    @property
    def K_A_ads(self):
        return jnp.exp(self._K_A_ads_inv)

    @property
    def K_A_des(self):
        return jnp.exp(self._K_A_des_inv)

    @property
    def K_B_ads(self):
        return jnp.exp(self._K_B_ads_inv)

    @property
    def K_B_des(self):
        return jnp.exp(self._K_B_des_inv)
