import jax.numpy as jnp
from jax.nn import sigmoid
from jax.scipy.special import logit

lower = 3 * logit(0.3)
upper = 3 * logit(0.7)

range = upper - lower
print(sigmoid(lower / 3), sigmoid(upper / 3))
print("alpha", range)
print("------")

lower = jnp.log10(0.1)
upper = jnp.log10(10_000)
range = upper - lower
print(jnp.power(10, lower), jnp.power(10, upper))
print("K0", range)
print("------")

lower = 5.0 * jnp.arctanh(-10.0 / 20.0)
upper = 5.0 * jnp.arctanh(10.0 / 20.0)
range = upper - lower
print(20.0 * jnp.tanh(lower / 5.0), 20.0 * jnp.tanh(upper / 5.0))
print("E0", range)
print("------")

lower = jnp.log2(0.1)
upper = jnp.log2(4.0)
range = upper - lower
print("dB", range)
print(jnp.power(2, lower), jnp.power(2, upper))
print("------")
