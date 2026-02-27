import jax.numpy as jnp
import pytest

from src.solvers import nonadiagonal_solve, pentadiagonal_solve, tridiagonal_solve


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


def test_nonadiagonal():
    N = 10

    d4l = -1 * jnp.ones((N - 4,))
    d3l = -1 * jnp.ones((N - 3,))
    d2l = -1 * jnp.ones((N - 2,))
    dl = -1 * jnp.ones((N - 1,))

    diag = 10 * jnp.ones((N,))

    du = -1 * jnp.ones((N - 1,))
    d2u = -1 * jnp.ones((N - 2,))
    d3u = -1 * jnp.ones((N - 3,))
    d4u = -1 * jnp.ones((N - 4,))

    A = (
        jnp.diag(diag)
        + jnp.diag(dl, k=-1)
        + jnp.diag(du, k=1)
        + jnp.diag(d2l, k=-2)
        + jnp.diag(d2u, k=2)
        + jnp.diag(d3l, k=-3)
        + jnp.diag(d3u, k=3)
        + jnp.diag(d4l, k=-4)
        + jnp.diag(d4u, k=4)
    )

    rhs = jnp.array([1.0, 4.0, 6.0, 8.0, 20.0, 27.0, 3.0, 11.0, 5.0, 9.0])

    solve_solution = jnp.linalg.solve(A, rhs)

    p_d4l = jnp.concatenate([jnp.zeros(4), d4l])
    p_d3l = jnp.concatenate([jnp.zeros(3), d3l])
    p_d2l = jnp.concatenate([jnp.zeros(2), d2l])
    p_dl = jnp.concatenate([jnp.zeros(1), dl])

    p_du = jnp.concatenate([du, jnp.zeros(1)])
    p_d2u = jnp.concatenate([d2u, jnp.zeros(2)])
    p_d3u = jnp.concatenate([d3u, jnp.zeros(3)])
    p_d4u = jnp.concatenate([d4u, jnp.zeros(4)])

    nonadiag_solution = nonadiagonal_solve(
        p_d4l, p_d3l, p_d2l, p_dl, diag, p_du, p_d2u, p_d3u, p_d4u, rhs
    )

    assert pytest.approx(solve_solution, rel=1e-4) == nonadiag_solution
