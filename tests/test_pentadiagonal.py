import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import grad

from src.solvers import pentadiagonal_solve


def test_pentadiagonal():
    N = 10

    d2l = -1 * jnp.ones((N - 2,))
    dl = -1 * jnp.ones((N - 1,))
    d = 6 * jnp.ones((N,))
    du = -1 * jnp.ones((N - 1,))
    d2u = -1 * jnp.ones((N - 2,))

    A = (
        jnp.diag(d)
        + jnp.diag(dl, k=-1)
        + jnp.diag(du, k=1)
        + jnp.diag(d2l, k=-2)
        + jnp.diag(d2u, k=2)
    )

    rhs = jnp.array([1.0, 4.0, 6.0, 8.0, 20.0, 27.0, 3.0, 11.0, 5.0, 9.0])

    solve_solution = jnp.linalg.solve(A, rhs)

    penta_solution = pentadiagonal_solve(d2l, dl, d, du, d2u, rhs)

    assert pytest.approx(solve_solution, rel=0.0001) == penta_solution


def test_pentadiagonal_vjp():
    key = jr.key(42)
    N = 50
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    e = jr.normal(k1, (N - 2,))
    a = jr.normal(k2, (N - 1,))
    b = jr.normal(k3, (N,)) + 7.0
    c = jr.normal(k4, (N - 1,))
    f = jr.normal(k5, (N - 2,))
    d = jr.normal(k6, (N,))

    def loss_penta(e, a, b, c, f, d):
        return jnp.sum(pentadiagonal_solve(e, a, b, c, f, d) ** 2)

    def loss_dense(e, a, b, c, f, d):
        A = (
            jnp.diag(b)
            + jnp.diag(a, -1)
            + jnp.diag(c, 1)
            + jnp.diag(e, -2)
            + jnp.diag(f, 2)
        )
        return jnp.sum(jnp.linalg.solve(A, d) ** 2)

    grads_penta = grad(loss_penta, argnums=(0, 1, 2, 3, 4, 5))(e, a, b, c, f, d)
    grads_dense = grad(loss_dense, argnums=(0, 1, 2, 3, 4, 5))(e, a, b, c, f, d)

    for i, (gp, gd) in enumerate(zip(grads_penta, grads_dense)):
        max_err = jnp.max(jnp.abs(gp - gd))
        assert max_err < 1e-5, f"arg {i}: max error = {max_err:.2e}"


def test_pentadiagonal_vjp_random_systems():
    for seed in range(10):
        key = jr.key(seed)
        N = 20 + seed * 5
        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
        e = jr.normal(k1, (N - 2,))
        a = jr.normal(k2, (N - 1,))
        b = jr.normal(k3, (N,)) + 7.0
        c = jr.normal(k4, (N - 1,))
        f = jr.normal(k5, (N - 2,))
        d = jr.normal(k6, (N,))

        def loss_penta(e, a, b, c, f, d):
            return jnp.sum(pentadiagonal_solve(e, a, b, c, f, d) ** 2)

        def loss_dense(e, a, b, c, f, d):
            A = (
                jnp.diag(b)
                + jnp.diag(a, -1)
                + jnp.diag(c, 1)
                + jnp.diag(e, -2)
                + jnp.diag(f, 2)
            )
            return jnp.sum(jnp.linalg.solve(A, d) ** 2)

        grads_penta = grad(loss_penta, argnums=(0, 1, 2, 3, 4, 5))(e, a, b, c, f, d)
        grads_dense = grad(loss_dense, argnums=(0, 1, 2, 3, 4, 5))(e, a, b, c, f, d)

        for i, (gp, gd) in enumerate(zip(grads_penta, grads_dense)):
            max_err = jnp.max(jnp.abs(gp - gd))
            assert max_err < 1e-4, f"seed {seed}, arg {i}: max error = {max_err:.2e}"
