import jax.numpy as jnp
from jax import block_until_ready, grad, jit

from src.fdm import ElectronReactionFDSolver, HeterogeneousReactionFDSolver
from src.reaction import ElectronReaction, HeterogeneousReaction
from src.voltammetry import CyclicDC

cyclic_dc = CyclicDC()

ele_solver = ElectronReactionFDSolver(cyclic_dc)
het_solver = HeterogeneousReactionFDSolver(cyclic_dc)
ele_params = ElectronReaction().true_parameters
het_params = HeterogeneousReaction().true_parameters

ele_fwd = jit(ele_solver.solve)
ele_grad = grad(lambda x: jnp.sum(ele_fwd(x)))

het_fwd = jit(het_solver.solve)
het_grad = grad(lambda x: jnp.sum(het_fwd(x)))

_ = block_until_ready(ele_fwd(ele_params))
_ = block_until_ready(ele_grad(ele_params))

_ = block_until_ready(het_fwd(het_params))
_ = block_until_ready(het_grad(het_params))

# %% Electron

ele_fwd(ele_params)
ele_grad(ele_params)

# %% Heterogeneous

het_fwd(het_params)
het_grad(het_params)
