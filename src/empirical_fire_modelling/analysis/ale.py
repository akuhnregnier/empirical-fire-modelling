# -*- coding: utf-8 -*-
"""ALE plots."""
import math
from collections import defaultdict
from functools import partial
from operator import attrgetter

import alepython.ale
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from alepython import ale_plot, multi_ale_plot_1d
from joblib import parallel_backend
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from wildfires.qstat import get_ncpus
from wildfires.utils import simple_sci_format

from ..cache import cache, process_proxy
from ..plotting import get_sci_format
from ..utils import column_check, tqdm

# Transparently cache the ALE computations.
alepython.ale.first_order_ale_quant = cache(alepython.ale.first_order_ale_quant)
alepython.ale.second_order_ale_quant = cache(
    alepython.ale.second_order_ale_quant, ignore=["n_jobs"]
)
alepython.ale._mc_replicas = cache(alepython.ale._mc_replicas, ignore=["verbose"])


def save_ale_1d(
    model,
    X_train,
    column,
    train_response=None,
    monte_carlo=True,
    monte_carlo_rep=100,
    monte_carlo_ratio=1000,
    monte_carlo_hull=True,
    verbose=True,
    center=False,
    figure_saver=None,
    sub_dir="ale",
    fig=None,
    ax=None,
):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(7.5, 4.5))
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    out = ale_plot(
        model,
        X_train,
        column,
        bins=20,
        train_response=train_response,
        monte_carlo=monte_carlo,
        monte_carlo_rep=monte_carlo_rep,
        monte_carlo_ratio=monte_carlo_ratio,
        monte_carlo_hull=monte_carlo_hull,
        plot_quantiles=True,
        quantile_axis=True,
        rugplot_lim=0,
        scilim=0.6,
        return_data=True,
        return_mc_data=True,
        verbose=verbose,
        center=center,
        rng=np.random.default_rng(0),
        fig=fig,
        ax=ax,
    )
    if monte_carlo:
        fig, axes, data, mc_data = out
    else:
        fig, axes, data = out

    for ax_key in ("ale", "quantiles_x"):
        axes[ax_key].xaxis.set_tick_params(rotation=45)

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
    features,
    train_response=None,
    monte_carlo=True,
    monte_carlo_rep=100,
    monte_carlo_ratio=1000,
    verbose=True,
    center=False,
    figure_saver=None,
    sub_dir="multi_ale",
    fig=None,
    ax=None,
    rngs=None,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 3)
        )  # Make sure plot is plotted onto a new figure.
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    if rngs is None:
        rngs = [np.random.default_rng(i) for i in range(len(features))]

    (
        fig,
        ax,
        final_quantiles,
        quantile_list,
        ale_list,
        mc_data_list,
    ) = multi_ale_plot_1d(
        model=model,
        train_set=X_train,
        features=features,
        train_response=train_response,
        monte_carlo=monte_carlo,
        monte_carlo_rep=monte_carlo_rep,
        monte_carlo_ratio=monte_carlo_ratio,
        return_data=True,
        return_mc_data=True,
        show_full=False,
        verbose=verbose,
        center=center,
        fig=fig,
        ax=ax,
        format_xlabels=False,
        xlabel_skip=1,
        rngs=rngs,
    )

    if figure_saver is not None:
        figure_saver.save_figure(
            fig, "__".join(map(str, features)), sub_directory=sub_dir
        )

    return final_quantiles


def get_model_predict(model):
    return model.predict


def single_ax_multi_ale_1d(
    ax,
    feature_data,
    feature,
    bins=20,
    xlabel=None,
    ylabel=None,
    title=None,
    verbose=False,
):
    quantile_list = []
    ale_list = []

    for experiment, single_experiment_data in zip(
        tqdm(
            feature_data["experiment"],
            desc="Calculating feature ALEs",
            disable=not verbose,
        ),
        feature_data["single_experiment_data"],
    ):
        model = single_experiment_data["model"]
        X_train = single_experiment_data["X_train"]

        with parallel_backend("threading", n_jobs=get_ncpus()):
            quantiles, ale = alepython.ale.first_order_ale_quant(
                process_proxy((model,), (get_model_predict,))[0],
                X_train,
                feature,
                bins=bins,
            )

        quantile_list.append(quantiles)
        ale_list.append(ale)

    # Construct quantiles from the individual quantiles, minimising the amount of interpolation.
    combined_quantiles = np.vstack([quantiles[None] for quantiles in quantile_list])

    final_quantiles = np.mean(combined_quantiles, axis=0)

    mod_quantiles = np.arange(len(quantiles))

    for plot_kwargs, quantiles, ale in zip(
        feature_data["plot_kwargs"], quantile_list, ale_list
    ):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            **{"marker": "o", "ms": 3, **plot_kwargs},
        )

        ax.set_xticks(mod_quantiles[::2])
        ax.set_xticklabels(
            map(
                lambda x: get_sci_format(ndigits=1, atol=np.inf)(x, None),
                final_quantiles[::2],
            )
        )
        ax.xaxis.set_tick_params(rotation=18)

        ax.grid(True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)


def multi_model_ale_1d(
    variable_factory,
    experiment_data,
    experiment_plot_kwargs,
    lags=(0, 1, 3, 6, 9),
    bins=20,
    title=None,
    verbose=False,
    figure_saver=None,
    single_figsize=(5.4, 1.5),
    legend_bbox=(0.5, 0.5),
    fig=None,
    axes=None,
    legend=True,
    y_ndigits=1,
):
    plotted_experiments = set()

    # Compile data for later plotting.
    comp_data = {}

    for lag in lags:
        assert lag <= 9

        feature = variable_factory[lag]

        feature_data = defaultdict(list)

        experiment_count = 0
        for experiment, single_experiment_data in experiment_data.items():
            # Skip experiments that do not contain this feature.
            if not column_check(single_experiment_data["X_train"], feature):
                continue

            experiment_count += 1
            plotted_experiments.add(experiment)

            # Data required to calculate the ALEs.
            feature_data["experiment"].append(experiment)
            feature_data["single_experiment_data"].append(single_experiment_data)
            feature_data["plot_kwargs"].append(experiment_plot_kwargs[experiment])

        if experiment_count <= 1:
            # We need at least two models for a comparison.
            continue

        comp_data[feature] = feature_data

    n_plots = len(comp_data)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    if fig is None and axes is None:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=np.array(single_figsize) * np.array([n_cols, n_rows]),
        )
    elif fig is not None and axes is not None:
        pass
    else:
        raise ValueError("Either both or none of fig and axes need to be given.")

    # Disable unused axes.
    if len(axes.flatten()) > n_plots:
        for ax in axes.flatten()[-(len(axes.flatten()) - n_plots) :]:
            ax.axis("off")

    for ax, feature, feature_data in zip(axes.flatten(), comp_data, comp_data.values()):
        single_ax_multi_ale_1d(
            ax,
            feature_data=feature_data,
            feature=feature,
            bins=bins,
            xlabel=str(feature),
            verbose=verbose,
        )

    for ax in axes.flatten()[:n_plots]:
        ax.yaxis.set_major_formatter(
            get_sci_format(ndigits=y_ndigits, atol_exceeded="adjust")
        )

    for row_axes in axes:
        row_axes[0].set_ylabel("ALE (BA)")

    fig.tight_layout()

    lines = []
    labels = []
    for experiment in sorted(plotted_experiments, key=attrgetter("value")):
        lines.append(Line2D([0], [0], **experiment_plot_kwargs[experiment]))
        labels.append(experiment_plot_kwargs[experiment]["label"])

    if legend:
        fig.legend(
            lines,
            labels,
            loc="center",
            bbox_to_anchor=legend_bbox,
            ncol=len(labels) if len(labels) <= 6 else 6,
        )

    if figure_saver is not None:
        figure_saver.save_figure(
            fig,
            f"{shorten_features(variable_factory).replace(' ', '_').lower()}_ale_comp",
            sub_directory="ale_comp",
        )
