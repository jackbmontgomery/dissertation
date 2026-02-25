import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.params import ECirreMechanismFDMParams, EMechanismFDMParams


def plot_e_histograms(datasets, heading: str):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))

    true_params = EMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(10.0),
        E0=jnp.array(2.0),
        dB=jnp.array(1.2),
    )

    hist_kwargs = dict(
        alpha=0.75,
        bins=50,
        density=True,
    )

    for d in datasets.keys():
        ax1.hist(datasets[d]["alpha"], label=d, **hist_kwargs)
        ax2.hist(datasets[d]["K0"], **hist_kwargs)
        ax3.hist(datasets[d]["E0"], **hist_kwargs)
        ax4.hist(datasets[d]["dB"], **hist_kwargs)

    ax1.set_title(r"$\alpha$")
    ax1.axvline(true_params.alpha, c="black", linestyle="--", label="True Value")
    ax2.set_title(r"$K_0$")
    ax2.axvline(true_params.K0, c="black", linestyle="--", label="True Value")
    ax3.set_title(r"$E_f^0$")
    ax3.axvline(true_params.E0, c="black", linestyle="--", label="True Value")
    ax4.set_title(r"$d_B$")
    ax4.axvline(true_params.dB, c="black", linestyle="--", label="True Value")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        heading,
        fontsize=18,
        y=0.98,
    )

    plt.show()

    print(f"{'Method':<8} {'Param':<10} {'μ ± σ':<30}")
    print("-" * 52)

    for method, params in datasets.items():
        for param, values in params.items():
            mu = jnp.mean(values)
            sigma = jnp.std(values)
            summary = f"{mu:.4e} ± {sigma:.4e}"
            print(f"{method:<8} {param:<10} {summary:<30}")
        print("-" * 28)
    print()


def plot_ec_irre_histograms(datasets, heading: str):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 5))

    true_params = ECirreMechanismFDMParams(
        alpha=jnp.array(0.6),
        K0=jnp.array(20.0),
        Kplus=jnp.array(10.0),
        Kminus=jnp.array(1.0),
        E0=jnp.array(2.0),
        dB=jnp.array(1.2),
    )

    hist_kwargs = dict(
        alpha=0.75,
        bins=50,
        density=True,
    )

    for d in datasets.keys():
        axs[0, 0].hist(datasets[d]["alpha"], label=d, **hist_kwargs)
        axs[0, 1].hist(datasets[d]["K0"], **hist_kwargs)
        axs[0, 2].hist(datasets[d]["E0"], **hist_kwargs)
        axs[1, 0].hist(datasets[d]["dB"], **hist_kwargs)
        axs[1, 1].hist(datasets[d]["Kplus"], **hist_kwargs)
        axs[1, 2].hist(datasets[d]["Kminus"], **hist_kwargs)

    axs[0, 0].set_title(r"$\alpha$")
    axs[0, 0].axvline(true_params.alpha, c="black", linestyle="--", label="True Value")
    axs[0, 1].set_title(r"$K_0$")
    axs[0, 1].axvline(true_params.K0, c="black", linestyle="--")
    axs[0, 2].set_title(r"$E_0$")
    axs[0, 2].axvline(true_params.E0, c="black", linestyle="--")

    axs[1, 0].set_title(r"$d_B$")
    axs[1, 0].axvline(true_params.dB, c="black", linestyle="--")
    axs[1, 1].set_title(r"$K_+$")
    axs[1, 1].axvline(true_params.Kplus, c="black", linestyle="--")
    axs[1, 2].set_title(r"$K_-$")
    axs[1, 2].axvline(true_params.Kminus, c="black", linestyle="--")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        heading,
        fontsize=18,
        y=0.98,
    )

    plt.tight_layout()
    plt.show()

    print(f"{'Method':<8} {'Param':<10} {'μ ± σ':<30}")
    print("-" * 52)

    for method, params in datasets.items():
        for param, values in params.items():
            mu = jnp.mean(values)
            sigma = jnp.std(values)
            summary = f"{mu:.4e} ± {sigma:.4e}"
            print(f"{method:<8} {param:<10} {summary:<30}")
        print("-" * 28)
    print()
