import jax.numpy as jnp
from chex import dataclass
from jax import tree_util
from jax.lax import scan
from jaxtyping import Scalar


@dataclass
class NonaMod:
    c: Scalar
    f: Scalar
    h: Scalar
    j: Scalar
    d: Scalar


@dataclass
class NonaFwdCarry:
    mod_m1: NonaMod
    mod_m2: NonaMod
    mod_m3: NonaMod
    mod_m4: NonaMod


@dataclass
class NonaBwdCarry:
    x_p1: Scalar
    x_p2: Scalar
    x_p3: Scalar
    x_p4: Scalar


@dataclass
class NonaRow:
    i: Scalar
    g: Scalar
    e: Scalar
    a: Scalar
    b: Scalar
    c: Scalar
    f: Scalar
    h: Scalar
    j: Scalar
    d: Scalar


def nonadiagonal_solve(
    i: Scalar,
    g: Scalar,
    e: Scalar,
    a: Scalar,
    b: Scalar,
    c: Scalar,
    f: Scalar,
    h: Scalar,
    j: Scalar,
    d: Scalar,
) -> Scalar:
    # NOTE:
    # i: 4th lower diagonal: first four elements are 0
    # g: 3rd lower diagonal: first three elements are 0
    # h: 3rd upper diagonal: last three elements are 0
    # j: 4th upper diagonal: last four elements are 0

    mod_0_denom = b[0]

    d_mod_0 = d[0] / mod_0_denom

    c_mod_0 = c[0] / mod_0_denom
    f_mod_0 = f[0] / mod_0_denom
    h_mod_0 = h[0] / mod_0_denom
    j_mod_0 = j[0] / mod_0_denom

    mod_0 = NonaMod(c=c_mod_0, f=f_mod_0, h=h_mod_0, j=j_mod_0, d=d_mod_0)

    coef_1 = a[1]

    mod_1_denom = b[1] - coef_1 * c_mod_0

    d_mod_1 = (d[1] - coef_1 * d_mod_0) / mod_1_denom

    c_mod_1 = (c[1] - coef_1 * f_mod_0) / mod_1_denom
    f_mod_1 = (f[1] - coef_1 * h_mod_0) / mod_1_denom
    h_mod_1 = (h[1] - coef_1 * j_mod_0) / mod_1_denom
    j_mod_1 = j[1] / mod_1_denom

    mod_1 = NonaMod(c=c_mod_1, f=f_mod_1, h=h_mod_1, j=j_mod_1, d=d_mod_1)

    coef_1 = e[2]
    coef_2 = a[2] - coef_1 * c_mod_0

    mod_2_denom = b[2] - coef_1 * f_mod_0 - coef_2 * c_mod_1

    d_mod_2 = (d[2] - coef_1 * d_mod_0 - coef_2 * d_mod_1) / mod_2_denom

    c_mod_2 = (c[2] - coef_1 * h_mod_0 - coef_2 * f_mod_1) / mod_2_denom
    f_mod_2 = (f[2] - coef_1 * j_mod_0 - coef_2 * h_mod_1) / mod_2_denom
    h_mod_2 = (h[2] - coef_2 * j_mod_1) / mod_2_denom
    j_mod_2 = j[2] / mod_2_denom

    mod_2 = NonaMod(c=c_mod_2, f=f_mod_2, h=h_mod_2, j=j_mod_2, d=d_mod_2)

    coef_1 = g[3]
    coef_2 = e[3] - coef_1 * c_mod_0
    coef_3 = a[3] - coef_1 * f_mod_0 - coef_2 * c_mod_1

    mod_3_denom = b[3] - coef_1 * h_mod_0 - coef_2 * f_mod_1 - coef_3 * c_mod_2

    d_mod_3 = (
        d[3] - coef_1 * d_mod_0 - coef_2 * d_mod_1 - coef_3 * d_mod_2
    ) / mod_3_denom

    c_mod_3 = (
        c[3] - coef_1 * j_mod_0 - coef_2 * h_mod_1 - coef_3 * f_mod_2
    ) / mod_3_denom

    f_mod_3 = (f[3] - coef_2 * j_mod_1 - coef_3 * h_mod_2) / mod_3_denom
    h_mod_3 = (h[3] - coef_3 * j_mod_2) / mod_3_denom
    j_mod_3 = j[3] / mod_3_denom

    mod_3 = NonaMod(c=c_mod_3, f=f_mod_3, h=h_mod_3, j=j_mod_3, d=d_mod_3)

    def fwd(carry: NonaFwdCarry, row: NonaRow):
        coef_1 = row.i
        coef_2 = row.g - coef_1 * carry.mod_m4.c
        coef_3 = row.e - coef_1 * carry.mod_m4.f - coef_2 * carry.mod_m3.c
        coef_4 = (
            row.a
            - coef_1 * carry.mod_m4.h
            - coef_2 * carry.mod_m3.f
            - coef_3 * carry.mod_m2.c
        )

        mod_denom = (
            row.b
            - coef_1 * carry.mod_m4.j
            - coef_2 * carry.mod_m3.h
            - coef_3 * carry.mod_m2.f
            - coef_4 * carry.mod_m1.c
        )

        d_mod = (
            row.d
            - coef_1 * carry.mod_m4.d
            - coef_2 * carry.mod_m3.d
            - coef_3 * carry.mod_m2.d
            - coef_4 * carry.mod_m1.d
        ) / mod_denom

        c_mod = (
            row.c
            - coef_2 * carry.mod_m3.j
            - coef_3 * carry.mod_m2.h
            - coef_4 * carry.mod_m1.f
        ) / mod_denom

        f_mod = (row.f - coef_3 * carry.mod_m2.j - coef_4 * carry.mod_m1.h) / mod_denom
        h_mod = (row.h - coef_4 * carry.mod_m1.j) / mod_denom
        j_mod = row.j / mod_denom

        mod = NonaMod(c=c_mod, f=f_mod, h=h_mod, j=j_mod, d=d_mod)

        new_carry = NonaFwdCarry(
            mod_m1=mod, mod_m2=carry.mod_m1, mod_m3=carry.mod_m2, mod_m4=carry.mod_m3
        )

        return new_carry, carry.mod_m4

    init_carry = NonaFwdCarry(mod_m1=mod_3, mod_m2=mod_2, mod_m3=mod_1, mod_m4=mod_0)
    xs = NonaRow(
        i=i[4:],
        g=g[4:],
        e=e[4:],
        a=a[4:],
        b=b[4:],
        c=c[4:],
        f=f[4:],
        h=h[4:],
        j=j[4:],
        d=d[4:],
    )
    carry, mods = scan(fwd, init_carry, xs)

    x_n = carry.mod_m1.d
    x_n_m1 = carry.mod_m2.d - carry.mod_m2.c * x_n
    x_n_m2 = carry.mod_m3.d - carry.mod_m3.c * x_n_m1 - carry.mod_m3.f * x_n
    x_n_m3 = (
        carry.mod_m4.d
        - carry.mod_m4.c * x_n_m2
        - carry.mod_m4.f * x_n_m1
        - carry.mod_m4.h * x_n
    )

    def bwd(carry: NonaBwdCarry, mod: NonaMod):
        x = (
            mod.d
            - mod.c * carry.x_p1
            - mod.f * carry.x_p2
            - mod.h * carry.x_p3
            - mod.j * carry.x_p4
        )

        new_carry = NonaBwdCarry(
            x_p1=x, x_p2=carry.x_p1, x_p3=carry.x_p2, x_p4=carry.x_p3
        )

        return new_carry, carry.x_p4

    init_carry = NonaBwdCarry(x_p1=x_n_m3, x_p2=x_n_m2, x_p3=x_n_m1, x_p4=x_n)
    xs = mods

    final_carry, x_last = scan(bwd, init_carry, xs, reverse=True)

    first_xs = jnp.array(tree_util.tree_leaves(final_carry))

    x = jnp.concat([first_xs, x_last])

    return x
