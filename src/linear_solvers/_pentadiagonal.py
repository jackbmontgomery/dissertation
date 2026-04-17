import ctypes
import os

import jax
import jax.ffi
from jax import custom_vjp, jit

_lib_path = os.path.join(os.path.dirname(__file__), "ffi", "penta_ffi.so")
_lib = ctypes.CDLL(_lib_path)

jax.ffi.register_ffi_target(
    "penta_solve_f64",
    jax.ffi.pycapsule(_lib.PentaSolveF64FFI),
    platform="cpu",
)
jax.ffi.register_ffi_target_as_batch_partitionable("penta_solve_f64")


@custom_vjp
def pentadiagonal_solve(e, a, b, c, f, d):
    return pentadiagonal_solve_impl(e, a, b, c, f, d)


def pentadiagonal_solve_fwd(e, a, b, c, f, d):
    x = pentadiagonal_solve_impl(e, a, b, c, f, d)
    return x, (e, a, b, c, f, x)


def pentadiagonal_solve_bwd(res, g):
    e, a, b, c, f, x = res
    lam = pentadiagonal_solve_impl(f, c, b, a, e, g)

    g_b = -lam * x
    g_a = -lam[1:] * x[:-1]
    g_c = -lam[:-1] * x[1:]
    g_e = -lam[2:] * x[:-2]
    g_f = -lam[:-2] * x[2:]
    g_d = lam

    return g_e, g_a, g_b, g_c, g_f, g_d


pentadiagonal_solve.defvjp(pentadiagonal_solve_fwd, pentadiagonal_solve_bwd)


@jit
def pentadiagonal_solve_impl(e, a, b, c, f, d):
    return jax.ffi.ffi_call(
        "penta_solve_f64",
        jax.ShapeDtypeStruct(d.shape, d.dtype),
        vmap_method="sequential",
    )(e, a, b, c, f, d)
