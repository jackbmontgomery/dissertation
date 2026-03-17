import jax.numpy as jnp
from equinox import Module
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Scalar

Params = Module


class ElectronReactionParams(Params):
    _alpha_inv: Scalar
    _K0_inv: Scalar
    _Ef_inv: Scalar

    def __init__(self, alpha: Scalar, K0: Scalar, Ef: Scalar):
        self._alpha_inv = 3.0 * logit(alpha)
        self._K0_inv = jnp.log2(K0)
        self._Ef_inv = 2.0 * jnp.arctanh(Ef / 10.0)

    @property
    def alpha(self):
        return sigmoid(self._alpha_inv / 3.0)

    @property
    def K0(self):
        return jnp.power(2, self._K0_inv)

    @property
    def Ef(self):
        return 10.0 * jnp.tanh(self._Ef_inv / 2.0)


class HeterogenousReactionParams(Params):
    _alpha_1_inv: Scalar
    _K0_1_inv: Scalar
    _Ef_1_inv: Scalar
    _alpha_2_inv: Scalar
    _K0_2_inv: Scalar
    _Ef_2_inv: Scalar
    _K_het_inv: Scalar

    def __init__(
        self,
        alpha_1: Scalar,
        K0_1: Scalar,
        Ef_1: Scalar,
        alpha_2: Scalar,
        K0_2: Scalar,
        Ef_2: Scalar,
        K_het: Scalar,
    ):
        self._alpha_1_inv = 3.0 * logit(alpha_1)
        self._K0_1_inv = jnp.log2(K0_1)
        self._Ef_1_inv = 5.0 * jnp.arctanh(Ef_1 / 20.0)

        self._alpha_2_inv = 3.0 * logit(alpha_2)
        self._K0_2_inv = jnp.log2(K0_2)
        self._Ef_2_inv = 5.0 * jnp.arctanh(Ef_2 / 20.0)

        self._K_het_inv = jnp.log2(K_het)

    @property
    def alpha_1(self):
        return sigmoid(self._alpha_1_inv / 3.0)

    @property
    def alpha_2(self):
        return sigmoid(self._alpha_2_inv / 3.0)

    @property
    def K0_1(self):
        return jnp.power(2, self._K0_1_inv)

    @property
    def K0_2(self):
        return jnp.power(2, self._K0_2_inv)

    @property
    def Ef_1(self):
        return 20.0 * jnp.tanh(self._Ef_1_inv / 5.0)

    @property
    def Ef_2(self):
        return 20.0 * jnp.tanh(self._Ef_2_inv / 5.0)

    @property
    def K_het(self):
        return jnp.power(2, self._K_het_inv)


class AdsorptionReactionParams(Params):
    _alpha_sol_inv: Scalar
    _K0_sol_inv: Scalar
    _Ef_sol_inv: Scalar
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
        Ef_sol: Scalar,
        alpha_ads: Scalar,
        K0_ads: Scalar,
        K_A_ads: Scalar,
        K_A_des: Scalar,
        K_B_ads: Scalar,
        K_B_des: Scalar,
    ):
        self._alpha_sol_inv = 3.0 * logit(alpha_sol)
        self._K0_sol_inv = jnp.log10(K0_sol)
        self._Ef_sol_inv = 5.0 * jnp.arctanh(Ef_sol / 20.0)

        self._alpha_ads_inv = 3.0 * logit(alpha_ads)
        self._K0_ads_inv = jnp.log10(K0_ads)

        self._K_A_ads_inv = jnp.log2(K_A_ads)
        self._K_A_des_inv = jnp.log2(K_A_des)
        self._K_B_ads_inv = jnp.log2(K_B_ads)
        self._K_B_des_inv = jnp.log2(K_B_des)

    @property
    def alpha_sol(self):
        return sigmoid(self._alpha_sol_inv / 3.0)

    @property
    def K0_sol(self):
        return jnp.power(10, self._K0_sol_inv)

    @property
    def Ef_sol(self):
        return 20.0 * jnp.tanh(self._Ef_sol_inv / 5.0)

    @property
    def alpha_ads(self):
        return sigmoid(self._alpha_ads_inv / 3.0)

    @property
    def K0_ads(self):
        return jnp.power(10, self._K0_ads_inv)

    @property
    def Ef_ads(self):
        return self.Ef_sol - jnp.log(
            (self.K_A_ads / self.K_A_des) / (self.K_B_ads / self.K_B_des)
        )

    @property
    def K_A_ads(self):
        return jnp.power(2, self._K_A_ads_inv)

    @property
    def K_A_des(self):
        return jnp.power(2, self._K_A_des_inv)

    @property
    def K_B_ads(self):
        return jnp.power(2, self._K_B_ads_inv)

    @property
    def K_B_des(self):
        return jnp.power(2, self._K_B_des_inv)
