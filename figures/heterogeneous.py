import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import HeterogeneousReactionSamplingExperiment

# %%

hmc = np.load("./data/H_DC_HMC_0.25_1000.npz")
rw = np.load("./data/H_RW_0.25_1000.npz")

true_params = HeterogeneousReactionSamplingExperiment().true_parameters

fig = plt.figure(figsize=(12, 5))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_K1 = fig.add_subplot(gs[0, 1])
ax_Ef1 = fig.add_subplot(gs[0, 2])
ax_a2 = fig.add_subplot(gs[1, 0])
ax_K2 = fig.add_subplot(gs[1, 1])
ax_Ef2 = fig.add_subplot(gs[1, 2])

gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0)
ax_Khet = fig.add_subplot(gs_right[1, 0])

options = {"density": True, "bins": 50, "alpha": 0.8, "histtype": "step"}

ax_a1.set_title(r"$\alpha^{(1)}$")
ax_a1.hist(hmc["alpha_1"].flatten(), label="HMC", **options)
ax_a1.hist(rw["alpha_1"].flatten(), **options, label="RW")
ax_a1.axvline(x=true_params.alpha_1, linestyle="--", color="black", label="True Value")

ax_a2.set_title(r"$\alpha^{(2)}$")
ax_a2.hist(hmc["alpha_2"].flatten(), **options)
ax_a2.hist(rw["alpha_2"].flatten(), **options)
ax_a2.axvline(x=true_params.alpha_2, linestyle="--", color="black")

ax_K1.set_title(r"$K_0^{(1)}$")
ax_K1.hist(hmc["K0_1"].flatten(), **options)
ax_K1.hist(rw["K0_1"].flatten(), **options)
ax_K1.axvline(x=true_params.K0_1, linestyle="--", color="black")

ax_K2.set_title(r"$K_0^{(2)}$")
ax_K2.hist(hmc["K0_2"].flatten(), **options)
ax_K2.hist(rw["K0_2"].flatten(), **options)
ax_K2.axvline(x=true_params.K0_2, linestyle="--", color="black")

ax_Ef1.set_title(r"$E_f^{(1)}$")
ax_Ef1.hist(hmc["Ef_1"].flatten(), **options)
ax_Ef1.hist(rw["Ef_1"].flatten(), **options)
ax_Ef1.axvline(x=true_params.Ef_1, linestyle="--", color="black")

ax_Ef2.set_title(r"$E_f^{(2)}$")
ax_Ef2.hist(hmc["Ef_2"].flatten(), **options)
ax_Ef2.hist(rw["Ef_2"].flatten(), **options)
ax_Ef2.axvline(x=true_params.Ef_2, linestyle="--", color="black")

ax_Khet.set_title(r"$K_{\text{het}}$")
ax_Khet.hist(hmc["K_het"].flatten(), **options)
ax_Khet.hist(rw["K_het"].flatten(), **options)
ax_Khet.axvline(x=true_params.K_het, linestyle="--", color="black")

handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1)

plt.show()
