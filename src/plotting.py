import inspect
import math

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.diagnostics import ess_over_time, potential_scale_reduction_over_time
from src.params import Params


def _property_names(params: Params) -> list[str]:
    return [
        name
        for name, val in inspect.getmembers(
            type(params), lambda v: isinstance(v, property)
        )
    ]


def plot_ess_over_time(
    samplers: dict[str, Params],
    num_points: int,
    labels: dict[str, str] | None = None,
    ignore_prop: list[str] = [],
) -> plt.Figure:
    """Plot ESS over the sampling run for all physical parameters.

    Args:
        samplers: Mapping of sampler name to pickled params module, e.g.
            ``{"NUTS": nuts_params, "RWMH": rwmh_params}``.
        num_points: Number of evenly-spaced points to evaluate ESS at.
        labels: Optional mapping of property name to display label (e.g. LaTeX
            string).  Defaults to the property name itself.

    Returns:
        The matplotlib Figure.
    """
    prop_names = [
        p
        for p in _property_names(next(iter(samplers.values())))
        if p not in ignore_prop
    ]
    n = len(prop_names)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axs_flat = axs.flatten() if n > 1 else [axs]

    x = jnp.linspace(1 / num_points, 1, num_points)

    for sampler_name, params in samplers.items():
        for i, prop in enumerate(prop_names):
            samples = getattr(params, prop)
            ess = ess_over_time(samples, num_points)
            axs_flat[i].plot(x, ess, label=sampler_name)
            title = labels.get(prop, prop) if labels else prop
            axs_flat[i].set_title(title)

    for ax in axs_flat[n:]:
        ax.set_visible(False)

    axs_flat[0].legend()
    for ax in axs_flat[(nrows - 1) * ncols : n]:
        ax.set_xlabel("Sample Proportion")
    for i in range(0, n, ncols):
        axs_flat[i].set_ylabel("ESS")

    fig.tight_layout()
    return fig


def plot_potential_scale_reduction_over_time(
    samplers: dict[str, Params],
    num_points: int,
    labels: dict[str, str] | None = None,
    ignore_prop: list[str] = [],
) -> plt.Figure:
    """Plot potential scale reduction (R-hat) over the sampling run for all
    physical parameters.

    Args:
        samplers: Mapping of sampler name to pickled params module, e.g.
            ``{"NUTS": nuts_params, "RWMH": rwmh_params}``.
        num_points: Number of evenly-spaced points to evaluate R-hat at.
        labels: Optional mapping of property name to display label (e.g. LaTeX
            string).  Defaults to the property name itself.

    Returns:
        The matplotlib Figure.
    """
    prop_names = [
        p
        for p in _property_names(next(iter(samplers.values())))
        if p not in ignore_prop
    ]
    n = len(prop_names)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axs_flat = axs.flatten() if n > 1 else [axs]

    x = jnp.linspace(1 / num_points, 1, num_points)

    for sampler_name, params in samplers.items():
        for i, prop in enumerate(prop_names):
            samples = getattr(params, prop)
            rhat = potential_scale_reduction_over_time(samples, num_points)
            axs_flat[i].plot(x, rhat, label=sampler_name)
            title = labels.get(prop, prop) if labels else prop
            axs_flat[i].set_title(title)

    for ax in axs_flat[n:]:
        ax.set_visible(False)

    axs_flat[0].legend()
    for ax in axs_flat[(nrows - 1) * ncols : n]:
        ax.set_xlabel("Sample Proportion")
    for i in range(0, n, ncols):
        axs_flat[i].set_ylabel("R-hat")

    fig.tight_layout()
    return fig
