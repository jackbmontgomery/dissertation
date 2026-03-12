import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import grad

from src.solvers import tridiagonal_solve


def test_tridiagonal():
    N = 10
    lower = -1 * jnp.ones((N - 1,))
    main = 4 * jnp.ones((N,))
    upper = -1 * jnp.ones((N - 1,))
    A = jnp.diag(main) + jnp.diag(lower, k=-1) + jnp.diag(upper, k=1)
    rhs = jnp.arange(1.0, N + 1.0)
    solve_solution = jnp.linalg.solve(A, rhs)

    print(lower.shape, main.shape, upper.shape, rhs.shape)
    tri_solution = tridiagonal_solve(lower, main, upper, rhs)
    assert pytest.approx(solve_solution, rel=0.0001) == tri_solution


def test_tridiagonal_vjp():
    key = jr.key(42)
    N = 50

    lower = jr.normal(key, (N - 1,))
    main = jr.normal(key, (N,)) + 5.0
    upper = jr.normal(key, (N - 1,))
    rhs = jr.normal(key, (N,))

    def loss_thomas(a, b, c, d):
        return jnp.sum(tridiagonal_solve(a, b, c, d) ** 2)

    def loss_dense(a, b, c, d):
        A = jnp.diag(b) + jnp.diag(a, -1) + jnp.diag(c, 1)
        return jnp.sum(jnp.linalg.solve(A, d) ** 2)

    grads_thomas = grad(loss_thomas, argnums=(0, 1, 2, 3))(lower, main, upper, rhs)
    grads_dense = grad(loss_dense, argnums=(0, 1, 2, 3))(lower, main, upper, rhs)

    for i, (gt, gd) in enumerate(zip(grads_thomas, grads_dense)):
        max_err = jnp.max(jnp.abs(gt - gd))
        assert max_err < 1e-5, f"arg {i}: max error = {max_err:.2e}"


def test_tridiagonal_vjp_random_systems():
    for seed in range(10):
        key = jr.key(seed)
        N = 20 + seed * 5
        k1, k2, k3, k4 = jr.split(key, 4)
        a = jr.normal(k1, (N - 1,))
        b = jr.normal(k2, (N,)) + 5.0
        c = jr.normal(k3, (N - 1,))
        d = jr.normal(k4, (N,))

        def loss_thomas(a, b, c, d):
            return jnp.sum(tridiagonal_solve(a, b, c, d) ** 2)

        def loss_dense(a, b, c, d):
            A = jnp.diag(b) + jnp.diag(a, -1) + jnp.diag(c, 1)
            return jnp.sum(jnp.linalg.solve(A, d) ** 2)

        grads_thomas = grad(loss_thomas, argnums=(0, 1, 2, 3))(a, b, c, d)
        grads_dense = grad(loss_dense, argnums=(0, 1, 2, 3))(a, b, c, d)

        for i, (gt, gd) in enumerate(zip(grads_thomas, grads_dense)):
            max_err = jnp.max(jnp.abs(gt - gd))
            assert max_err < 1e-4, f"seed {seed}, arg {i}: max error = {max_err:.2e}"
