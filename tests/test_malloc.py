import tracemalloc

import jax
import jax.numpy as jnp

from src.fdm import ElectronReactionFDSolver
from src.params import ElectronReactionParams
from src.voltammetry import CyclicDC

for h0 in [1e-7, 1e-5, 1e-3]:
    for dtheta in [0.01, 0.05, 0.1]:
        jax.clear_caches()

        voltammetry = CyclicDC()
        fdm_solver = ElectronReactionFDSolver(voltammetry, h0=h0, dtheta=dtheta)

        params = ElectronReactionParams(
            alpha=jnp.array(0.6),
            K0=jnp.array(10.0),
            Ef=jnp.array(0.5),
        )

        _, true_current = fdm_solver.solve(params)

        def loss_fn(params):
            _, pred_current = fdm_solver.solve(params)
            return jnp.square(jnp.mean(pred_current - true_current))

        # Measure gradient computation
        tracemalloc.start()
        tracemalloc.reset_peak()

        grads = jax.grad(loss_fn)(params)
        jax.block_until_ready(grads)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"h0={h0}, dtheta={dtheta}, peak={peak / 1024 / 1024:.1f} MB")
