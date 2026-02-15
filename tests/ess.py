import blackjax
import jax.numpy as jnp
import numpy as np

nuts = np.load("./data/E_Nuts_LinearSweepDC.npz")
mh = np.load("./data/E_MetropolisHastings_LinearSweepDC.npz")
path = np.load("./data/E_Pathfinder_LinearSweepDC.npz")

nuts_alpha_samples = jnp.array([nuts["alpha"]])
nuts_K0_samples = jnp.array([nuts["K0"]])
mh_alpha_samples = jnp.array([mh["alpha"]])
mh_K0_samples = jnp.array([mh["K0"]])
path_alpha_samples = jnp.array([path["alpha"]])
path_K0_samples = jnp.array([path["K0"]])

print("Nuts Alpha", blackjax.diagnostics.effective_sample_size(nuts_alpha_samples))
print("Nuts K0", blackjax.diagnostics.effective_sample_size(nuts_K0_samples))

print("MH Alpha", blackjax.diagnostics.effective_sample_size(mh_alpha_samples))
print("MH K0", blackjax.diagnostics.effective_sample_size(mh_K0_samples))

print(
    "Pathfinder Alpha", blackjax.diagnostics.effective_sample_size(path_alpha_samples)
)
print("Pathfinder K0", blackjax.diagnostics.effective_sample_size(path_K0_samples))
