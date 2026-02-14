import jax.numpy as jnp
from chex import dataclass
from jax.lax import scan
from jaxtyping import Scalar


@dataclass
class TriMod:
    c: Scalar
    d: Scalar


@dataclass
class TriRow:
    a: Scalar
    b: Scalar
    c: Scalar
    d: Scalar


def tridiagonal_solve(
    a: Scalar,
    b: Scalar,
    c: Scalar,
    d: Scalar,
) -> Scalar:
    # NOTE: first entry in a is 0 and last entry in c is 0
    mod_1_denom = b[0]
    c_mod_1 = c[0] / mod_1_denom
    d_mod_1 = d[0] / mod_1_denom

    def fwd(mod: TriMod, row: TriRow):
        mod_denom = row.b - row.a * mod.c

        c_mod = row.c / mod_denom
        d_mod = (row.d - row.a * mod.d) / mod_denom

        new_mod = TriMod(c=c_mod, d=d_mod)
        return new_mod, mod

    init_mod = TriMod(c=c_mod_1, d=d_mod_1)
    xs = TriRow(a=a[1:], b=b[1:], c=c[1:], d=d[1:])

    final_mod, mods = scan(fwd, init_mod, xs)

    def bwd(x_p1: Scalar, mod: TriMod):
        x = mod.d - mod.c * x_p1
        return x, x

    x_n = final_mod.d
    _, x = scan(bwd, x_n, mods, reverse=True)

    solution = jnp.concat([x, jnp.array([x_n])])

    return solution
