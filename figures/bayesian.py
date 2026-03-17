import gzip
import pickle

import blackjax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax._src.flatten_util import unravel_pytree
from jax.flatten_util import ravel_pytree

from src.params import ElectronReactionParams
from src.reaction import ElectronReaction

DIR = "./data/sampling"

# %%

with gzip.open(f"{DIR}/reaction=ElectronReaction,noise=0.25,seed=0.pkl.gz", "rb") as f:
    data = pickle.load(f)

hmc: ElectronReactionParams = data["hmc"]
rwmh: ElectronReactionParams = data["rwmh"]

plt.hist(hmc.alpha.flatten(), label="HMC", density=True)
plt.hist(rwmh.alpha.flatten(), label="RWMH", density=True)
plt.legend()
plt.show()
