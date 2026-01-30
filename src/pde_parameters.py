import jax.nn as jnn
import jax.numpy as jnp
from equinox import Module
from jaxtyping import Scalar

AbstractPDEParameters = Module


class ButlerVolmerInverseParameters(AbstractPDEParameters):
    a: Scalar
    k0: Scalar


class ButlerVolmerPhysicalParameters(AbstractPDEParameters):
    alpha: Scalar
    kappa0: Scalar


def bv_inverse_to_physical(
    inv: ButlerVolmerInverseParameters,
) -> ButlerVolmerPhysicalParameters:
    alpha = jnn.sigmoid(inv.a)
    kappa0 = jnp.exp(inv.k0)
    phy = ButlerVolmerPhysicalParameters(alpha=alpha, kappa0=kappa0)
    return phy
