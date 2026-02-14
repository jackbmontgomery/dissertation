import jax.numpy as jnp
import pytest

from src.solvers import pentadiagonal_solve, tridiagonal_solve


def test_tridiagonal():
    N = 10

    lower = -1 * jnp.ones((N - 1,))
    main = 4 * jnp.ones((N,))
    upper = -1 * jnp.ones((N - 1,))

    A = jnp.diag(main) + jnp.diag(lower, k=-1) + jnp.diag(upper, k=1)

    rhs = jnp.arange(1.0, N + 1.0)

    solve_solution = jnp.linalg.solve(A, rhs)

    a = jnp.concat([jnp.array([0.0]), lower])
    b = main
    c = jnp.concat([upper, jnp.array([0.0])])
    d = rhs

    tri_solution = tridiagonal_solve(a, b, c, d)

    assert pytest.approx(solve_solution, rel=0.0001) == tri_solution


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

    e = jnp.concat([jnp.array([0.0, 0.0]), d2l])
    a = jnp.concat([jnp.array([0.0]), dl])
    b = d
    c = jnp.concat([du, jnp.array([0.0])])
    f = jnp.concat([d2u, jnp.array([0.0, 0.0])])
    d = rhs

    penta_solution = pentadiagonal_solve(e, a, b, c, f, d)

    assert pytest.approx(solve_solution, rel=0.0001) == penta_solution
