import corner
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use("science")

mchmc = np.load("./data/E_MCHMC_CyclicDC.npz")
rw = np.load("./data/E_MetropolisHastings_CyclicDC.npz")

alpha = rw["alpha"].flatten()
K0 = rw["K0"].flatten()
Ef = rw["E0"].flatten()
dB = rw["dB"].flatten()

data = np.stack([alpha, K0, Ef, dB], axis=1)
print(data.shape)
corner.corner(data)
plt.show()
