import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mchmc = np.load("./data/E_MCHMC_CyclicDC.npz")
rw = np.load("./data/E_MetropolisHastings_CyclicDC.npz")

df_mchmc = pd.DataFrame({k: mchmc[k].flatten() for k in mchmc.files})
df_rw = pd.DataFrame({k: rw[k].flatten() for k in rw.files})

df_mchmc["algorithm"] = "MCHMC"
df_rw["algorithm"] = "RW"

df_mchmc["algorithm"] = df_mchmc["algorithm"].astype("category")
df_rw["algorithm"] = df_rw["algorithm"].astype("category")

sampling_df = pd.concat([df_mchmc, df_rw], ignore_index=True)

g = sns.PairGrid(
    sampling_df.drop(columns=["logdensity"]),
    corner=True,
    despine=False,
    layout_pad=True,
    hue="algorithm",
)

g.map_diag(sns.histplot)
g.map_lower(sns.kdeplot)


plt.show()
