import numpy as np

from src.plotting import plot_ec_irre_histograms

mchmc = np.load("./data/ECirre_MCHMC_CyclicDC.npz")
rw = np.load("./data/ECirre_MetropolisHastings_CyclicDC.npz")

datasets = {
    "MCHMC": {
        "alpha": mchmc["alpha"].flatten(),
        "K0": mchmc["K0"].flatten(),
        "E0": mchmc["E0"].flatten(),
        "Kplus": mchmc["Kplus"].flatten(),
        "Kminus": mchmc["Kminus"].flatten(),
        "dB": mchmc["dB"].flatten(),
    },
    "RW": {
        "alpha": rw["alpha"].flatten(),
        "K0": rw["K0"].flatten(),
        "Kplus": rw["Kplus"].flatten(),
        "Kminus": rw["Kminus"].flatten(),
        "E0": rw["E0"].flatten(),
        "dB": rw["dB"].flatten(),
    },
}

plot_ec_irre_histograms(
    datasets,
    heading="Posterior Distribution for Electrode-only Reaction using Cyclic Voltammetry",
)
