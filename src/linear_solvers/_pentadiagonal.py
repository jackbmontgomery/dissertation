import jax.numpy as jnp
from chex import dataclass
from jax import custom_vjp, jit, tree_util
from jax.lax import scan
from jaxtyping import Scalar


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


@dataclass
class PentaMod:
    c: Scalar
    f: Scalar
    d: Scalar


@dataclass
class PentaFwdCarry:
    mod_m1: PentaMod
    mod_m2: PentaMod


@dataclass
class PentaBwdCarry:
    x_p1: Scalar
    x_p2: Scalar


@dataclass
class PentaRow:
    e: Scalar
    a: Scalar
    b: Scalar
    c: Scalar
    f: Scalar
    d: Scalar


@jit
def pentadiagonal_solve_impl(
    e: Scalar,
    a: Scalar,
    b: Scalar,
    c: Scalar,
    f: Scalar,
    d: Scalar,
) -> Scalar:
    mod_1_denom = b[0]

    c_mod_1 = c[0] / mod_1_denom
    f_mod_1 = f[0] / mod_1_denom
    d_mod_1 = d[0] / mod_1_denom

    mod_1 = PentaMod(c=c_mod_1, f=f_mod_1, d=d_mod_1)

    mod_2_denom = b[1] - a[0] * c_mod_1

    c_mod_2 = (c[1] - a[0] * f_mod_1) / mod_2_denom
    f_mod_2 = f[1] / mod_2_denom
    d_mod_2 = (d[1] - a[0] * d_mod_1) / mod_2_denom

    mod_2 = PentaMod(c=c_mod_2, f=f_mod_2, d=d_mod_2)

    def fwd(carry: PentaFwdCarry, row: PentaRow):
        x = row.a - row.e * carry.mod_m2.c
        inv_mod_denom = 1.0 / (row.b - row.e * carry.mod_m2.f - x * carry.mod_m1.c)

        c_mod = (row.c - x * carry.mod_m1.f) * inv_mod_denom
        f_mod = row.f * inv_mod_denom
        d_mod = (row.d - row.e * carry.mod_m2.d - x * carry.mod_m1.d) * inv_mod_denom

        mod = PentaMod(c=c_mod, f=f_mod, d=d_mod)
        new_carry = PentaFwdCarry(mod_m1=mod, mod_m2=carry.mod_m1)

        return new_carry, carry.mod_m2

    init_carry = PentaFwdCarry(mod_m1=mod_2, mod_m2=mod_1)

    xs = PentaRow(
        e=e,
        a=a[1:],
        b=b[2:],
        c=jnp.concat([c[2:], jnp.zeros(1)]),
        f=jnp.concat([f[2:], jnp.zeros(2)]),
        d=d[2:],
    )

    carry, mods = scan(fwd, init_carry, xs)

    def bwd(carry: PentaBwdCarry, mod: PentaMod):
        x = mod.d - mod.c * carry.x_p1 - mod.f * carry.x_p2
        new_carry = PentaBwdCarry(x_p1=x, x_p2=carry.x_p1)
        return new_carry, carry.x_p2

    x_n = carry.mod_m1.d
    x_n_m1 = carry.mod_m2.d - carry.mod_m2.c * x_n

    init_carry = PentaBwdCarry(x_p1=x_n_m1, x_p2=x_n)
    xs = mods

    final_carry, x = scan(bwd, init_carry, mods, reverse=True)

    first_xs = jnp.array(tree_util.tree_leaves(final_carry))

    solution = jnp.concat([first_xs, x])

    return solution
