import matplotlib.pyplot as plt

from src.fdm import HeterogeneousReactionFDSolver
from src.reaction import HeterogeneousReaction
from src.voltammetry import CyclicDC

cyclic_dc = CyclicDC()
fd_solver = HeterogeneousReactionFDSolver(cyclic_dc)
params = HeterogeneousReaction().true_parameters
sol = fd_solver.solve(params).block_until_ready()

plt.plot(fd_solver.applied_potentials, sol)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# %%

fd_solver.solve(params).block_until_ready()
