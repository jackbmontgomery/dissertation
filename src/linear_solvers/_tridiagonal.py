import ctypes
import os

import jax
import jax.ffi
from jax import custom_vjp, jit

_lib_path = os.path.join(os.path.dirname(__file__), "ffi", "tri_ffi.so")
_lib = ctypes.CDLL(_lib_path)

jax.ffi.register_ffi_target(
    "tri_solve_f64",
    jax.ffi.pycapsule(_lib.TriSolveF64FFI),
    platform="cpu",
)
jax.ffi.register_ffi_target_as_batch_partitionable("tri_solve_f64")


@custom_vjp
def tridiagonal_solve(a, b, c, d):
    return tridiagonal_solve_impl(a, b, c, d)


def tridiagonal_solve_fwd(a, b, c, d):
    x = tridiagonal_solve_impl(a, b, c, d)
    return x, (a, b, c, x)


def tridiagonal_solve_bwd(res, g):
    a, b, c, x = res
    lam = tridiagonal_solve_impl(c, b, a, g)
    g_b = -lam * x
    g_a = -lam[1:] * x[:-1]
    g_c = -lam[:-1] * x[1:]
    g_d = lam
    return g_a, g_b, g_c, g_d


tridiagonal_solve.defvjp(tridiagonal_solve_fwd, tridiagonal_solve_bwd)


@jit
def tridiagonal_solve_impl(a, b, c, d):
    return jax.ffi.ffi_call(
        "tri_solve_f64",
        jax.ShapeDtypeStruct(d.shape, d.dtype),
        vmap_method="sequential",
    )(a, b, c, d)
