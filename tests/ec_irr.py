import jax.numpy as jnp

T = jnp.linspace(0, 10.0, 4)
X = jnp.linspace(0, 5.0, 4)
N = len(X)

k0 = 100.0
km = 10.0
kp = 12.0

dt0 = T[1] - T[0]
dl2 = jnp.repeat(-dt0 / (X[1:-1] - X[:-2]) ** 2, 2)
du2 = jnp.repeat(-dt0 / (X[1:-1] - X[:-2]) ** 2, 2)

d = jnp.ones((2 * N - 2,))

dlB = jnp.full((N - 1,), -1 * dt0 * km)
dl = jnp.zeros((2 * N - 3,))
dl = dl.at[::2].set(dlB)

duA = jnp.full((N - 2,), -1 * dt0 * kp)
du = jnp.zeros((2 * N - 3,))
du = du.at[1::2].set(duA)

print(d.shape)
print(dl.shape)
print(dl2.shape)

A = (
    jnp.diag(d)
    + jnp.diag(dl, k=-1)
    + jnp.diag(dl2, k=-2)
    + jnp.diag(du, k=1)
    + jnp.diag(du2, k=2)
)
print(A)
