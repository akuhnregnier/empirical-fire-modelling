# -*- coding: utf-8 -*-
"""ALE plots."""

from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from alepython.ale import _sci_format, ale_plot, first_order_ale_quant
from joblib import parallel_backend
from matplotlib.colors import SymLogNorm
from wildfires.utils import NoCachedDataError, SimpleCache, simple_sci_format, tqdm


def save_ale_1d(
    model,
    X_train,
    column,
    n_jobs=8,
    monte_carlo_rep=1000,
    monte_carlo_ratio=100,
    verbose=False,
    monte_carlo=True,
    center=False,
    figure_saver=None,
):
    model.n_jobs = n_jobs
    with parallel_backend("threading", n_jobs=n_jobs):
        fig, ax = plt.subplots(
            figsize=(7.5, 4.5)
        )  # Make sure plot is plotted onto a new figure.
        out = ale_plot(
            model,
            X_train,
            column,
            bins=20,
            monte_carlo=monte_carlo,
            monte_carlo_rep=monte_carlo_rep,
            monte_carlo_ratio=monte_carlo_ratio,
            plot_quantiles=True,
            quantile_axis=True,
            rugplot_lim=0,
            scilim=0.6,
            return_data=True,
            return_mc_data=True,
            verbose=verbose,
            center=center,
        )
    if monte_carlo:
        fig, axes, data, mc_data = out
    else:
        fig, axes, data = out

    for ax_key in ("ale", "quantiles_x"):
        axes[ax_key].xaxis.set_tick_params(rotation=45)

    sub_dir = "ale" if monte_carlo else "ale_non_mc"
    if figure_saver is not None:
        figure_saver.save_figure(fig, str(column), sub_directory=sub_dir)


def save_ale_2d(
    model,
    train_set,
    features,
    n_jobs=1,
    include_first_order=True,
    figure_saver=None,
    plot_samples=True,
    figsize=(6.15, 3.17),
):
    model.n_jobs = n_jobs

    cbar_width = 0.01

    x_coords = {}
    x_coords["ALE start"] = 0
    x_coords["ALE end"] = 0.42
    x_coords["ALE cbar start"] = x_coords["ALE end"] + 0.01
    x_coords["ALE cbar end"] = x_coords["ALE cbar start"] + cbar_width
    x_coords["Samples start"] = 0.65
    x_coords["Samples end"] = 0.9
    x_coords["Samples cbar start"] = x_coords["Samples end"] + 0.01
    x_coords["Samples cbar end"] = x_coords["Samples cbar start"] + cbar_width

    y_bottom = {
        "Samples": 1 / 3,  # Samples plot and cbar bottom.
    }
    cbar_height = {
        "ALE": 0.6,
        "Samples": 0.4,
    }

    top = 1

    fig = plt.figure(figsize=figsize)

    # ALE plot axes.
    ax = [
        fig.add_axes(
            [x_coords["ALE start"], 0, x_coords["ALE end"] - x_coords["ALE start"], top]
        )
    ]
    # ALE plot cbar axes.
    cax = [
        fig.add_axes(
            [
                x_coords["ALE cbar start"],
                top * (1 - cbar_height["ALE"]) / 2,
                x_coords["ALE cbar end"] - x_coords["ALE cbar start"],
                cbar_height["ALE"],
            ]
        )
    ]
    if plot_samples:
        # Samples plot axes.
        ax.append(
            fig.add_axes(
                [
                    x_coords["Samples start"],
                    y_bottom["Samples"],
                    x_coords["Samples end"] - x_coords["Samples start"],
                    top - y_bottom["Samples"],
                ]
            )
        )
        # Samples plot cbar axes.
        cax.append(
            fig.add_axes(
                [
                    x_coords["Samples cbar start"],
                    (y_bottom["Samples"] + top) / 2 - cbar_height["Samples"] / 2,
                    x_coords["Samples cbar end"] - x_coords["Samples cbar start"],
                    cbar_height["Samples"],
                ]
            )
        )

    with parallel_backend("threading", n_jobs=n_jobs):
        fig, axes, (quantiles_list, ale, samples) = ale_plot(
            model,
            train_set,
            features,
            bins=20,
            fig=fig,
            ax=ax[0],
            plot_quantiles=False,
            quantile_axis=True,
            plot_kwargs={
                "kind": "grid",
                "cmap": "inferno",
                "colorbar_kwargs": dict(
                    format=ticker.FuncFormatter(
                        lambda x, pos: simple_sci_format(x, precision=1)
                    ),
                    cax=cax[0],
                    label="ALE (BA)",
                ),
            },
            return_data=True,
            n_jobs=n_jobs,
            include_first_order=include_first_order,
        )

    for ax_key in ("ale", "quantiles_x"):
        if ax_key in axes:
            axes[ax_key].xaxis.set_tick_params(rotation=50)

    axes["ale"].set_aspect("equal")
    axes["ale"].set_xlabel(features[0].units)
    axes["ale"].set_ylabel(features[1].units)
    axes["ale"].set_title("")

    axes["ale"].xaxis.set_ticklabels(
        np.vectorize(partial(simple_sci_format, precision=1))(quantiles_list[0])
    )
    axes["ale"].yaxis.set_ticklabels(
        np.vectorize(partial(simple_sci_format, precision=1))(quantiles_list[1])
    )

    for tick in axes["ale"].xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment("right")

    if plot_samples:
        # Plotting samples.
        mod_quantiles_list = []
        for axis, quantiles in zip(("x", "y"), quantiles_list):
            inds = np.arange(len(quantiles))
            mod_quantiles_list.append(inds)
            ax[1].set(**{f"{axis}ticks": inds})
            ax[1].set(
                **{
                    f"{axis}ticklabels": np.vectorize(
                        partial(simple_sci_format, precision=1)
                    )(quantiles)
                }
            )
            for label in getattr(ax[1], f"{axis}axis").get_ticklabels()[1::2]:
                label.set_visible(False)

        samples_img = ax[1].pcolormesh(
            *mod_quantiles_list, samples.T, norm=SymLogNorm(linthresh=1)
        )

        @ticker.FuncFormatter
        def samples_colorbar_fmt(x, pos):
            if x < 0:
                raise ValueError("Samples cannot be -ve.")
            if np.isclose(x, 0):
                return "0"
            if np.log10(x).is_integer():
                return simple_sci_format(x)
            return ""

        fig.colorbar(
            samples_img,
            cax=cax[1],
            label="samples",
            format=samples_colorbar_fmt,
        )
        ax[1].xaxis.set_tick_params(rotation=50)
        for tick in ax[1].xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment("right")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel(features[0].units)
        ax[1].set_ylabel(features[1].units)
        fig.set_constrained_layout_pads(
            w_pad=0.000, h_pad=0.000, hspace=0.0, wspace=0.015
        )

    if figure_saver is not None:
        if plot_samples:
            figure_saver.save_figure(
                fig,
                "__".join(map(str, features)),
                sub_directory="2d_ale_first_order" if include_first_order else "2d_ale",
            )
        else:
            figure_saver.save_figure(
                fig,
                "__".join(map(str, features)) + "_no_count",
                sub_directory="2d_ale_first_order_no_count"
                if include_first_order
                else "2d_ale_no_count",
            )


def multi_ale_1d(
    model,
    X_train,
    columns,
    fig_name=None,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    n_jobs=1,
    verbose=False,
    figure_saver=None,
    CACHE_DIR=None,
    bins=20,
    x_rotation=20,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 3)
        )  # Make sure plot is plotted onto a new figure.
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    quantile_list = []
    ale_list = []
    for feature in tqdm(columns, desc="Calculating feature ALEs", disable=not verbose):
        cache = SimpleCache(
            f"{feature}_ale_{bins}",
            cache_dir=CACHE_DIR / "ale",
            verbose=10 if verbose else 0,
        )
        try:
            quantiles, ale = cache.load()
        except NoCachedDataError:
            model.n_jobs = n_jobs

            with parallel_backend("threading", n_jobs=n_jobs):
                quantiles, ale = first_order_ale_quant(
                    model.predict, X_train, feature, bins=bins
                )
                cache.save((quantiles, ale))

        quantile_list.append(quantiles)
        ale_list.append(ale)

    # Construct quantiles from the individual quantiles, minimising the amount of interpolation.
    combined_quantiles = np.vstack([quantiles[None] for quantiles in quantile_list])

    final_quantiles = np.mean(combined_quantiles, axis=0)

    mod_quantiles = np.arange(len(quantiles))

    markers = ["o", "v", "^", "<", ">", "x", "+"]
    for feature, quantiles, ale, marker in zip(
        columns, quantile_list, ale_list, markers
    ):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            marker=marker,
            label=feature,
        )

    ax.legend(loc="best", ncol=2)

    ax.set_xticks(mod_quantiles[::2])
    ax.set_xticklabels(_sci_format(final_quantiles[::2], scilim=0.6))
    ax.xaxis.set_tick_params(rotation=x_rotation)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))
    )

    fig.suptitle(title)
    ax.set_xlabel(xlabel, va="center_baseline")
    ax.set_ylabel(ylabel)

    if figure_saver is not None:
        figure_saver.save_figure(fig, fig_name, sub_directory="multi_ale")
