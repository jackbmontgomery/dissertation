import numpy as np

from src.plotting import plot_e_histograms

mchmc = np.load("./data/E_MCHMC_CyclicDC.npz")
rw = np.load("./data/E_MetropolisHastings_CyclicDC.npz")

datasets = {
    "MCHMC": {
        "alpha": mchmc["alpha"].flatten(),
        "K0": mchmc["K0"].flatten(),
        "E0": mchmc["E0"].flatten(),
        "dB": mchmc["dB"].flatten(),
    },
    "RW": {
        "alpha": rw["alpha"].flatten(),
        "K0": rw["K0"].flatten(),
        "E0": rw["E0"].flatten(),
        "dB": rw["dB"].flatten(),
    },
}

plot_e_histograms(
    datasets,
    heading="Posterior Distribution for Electrode-only Reaction using Cyclic Voltammetry",
)
