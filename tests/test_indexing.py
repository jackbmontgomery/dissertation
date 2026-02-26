import jax.numpy as jnp

from src.utils import interleave_concat_2d

Nx = 10

A = jnp.full((Nx,), 1.0)
B = jnp.full((Nx,), 0.0)
C = jnp.full((Nx,), 0.1)
D = jnp.full((Nx,), 0.2)

X = jnp.concat([interleave_concat_2d(A, B), interleave_concat_2d(C, D)])

print(X.shape)
print(X)
A_inner = X[2 : 2 * Nx - 2 : 2]
B_inner = X[3 : 2 * Nx - 2 : 2]
C_inner = X[2 * Nx + 2 : -2 : 2]
D_inner = X[2 * Nx + 3 : -2 : 2]
print("A_inner", A_inner.shape)
print("B_inner", B_inner.shape)
print("C_inner", C_inner.shape)
print("D_inner", D_inner.shape)
