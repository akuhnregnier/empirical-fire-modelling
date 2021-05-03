# -*- coding: utf-8 -*-
"""ALE plots."""
import math
from collections import defaultdict
from copy import deepcopy
from operator import attrgetter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sklearn.base
from joblib import parallel_backend
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from wildfires.cache.proxy_backend import HashProxy
from wildfires.qstat import get_ncpus
from wildfires.utils import shorten_features

from .. import variable
from ..cache import cache, get_proxied_estimator, process_proxy
from ..plotting import get_float_format, get_sci_format, update_label_with_exp
from ..utils import column_check, tqdm


@cache
def cached_clone(estimator, safe=None):
    """Adapted from sklearn.base.clone."""
    estimator_params = estimator.get_params(deep=False)
    new_estimator = estimator.__class__(**deepcopy(estimator_params))
    new_params = new_estimator.get_params(deep=False)
    for name in estimator_params:
        param1 = estimator_params[name]
        param2 = new_params[name]
        assert param1 is param2
    return new_estimator


# Transparently cache estimator cloning.
# sklearn.base.clone = cache(sklearn.base.clone)
sklearn.base.clone = cached_clone


# Import after the line above in order for the `clone` caching to take effect within
# the module.
import alepython.ale  # isort:skip


# Transparently cache the ALE computations.
alepython.ale.first_order_ale_quant = cache(alepython.ale.first_order_ale_quant)
alepython.ale.second_order_ale_quant = cache(
    alepython.ale.second_order_ale_quant, ignore=["n_jobs"]
)
alepython.ale._mc_replicas = cache(alepython.ale._mc_replicas, ignore=["verbose"])

# Make use of proxied access to `predict` implicitly.
_orig_ale_plot = alepython.ale.ale_plot


def proxied_ale_plot(**kwargs):
    kwargs["model"] = get_proxied_estimator(kwargs["model"])
    return _orig_ale_plot(**kwargs)


alepython.ale.ale_plot = proxied_ale_plot


# Implicitly handle getting array data from Series.
orig_asarray = np.asarray


def lazy_series_asarray(*args, **kwargs):
    if len(args) == 1 and not kwargs and isinstance(args[0], HashProxy):
        # Handle the specific case of a single lazy input argument to avoid
        # loading it from disk for as long as possible.
        return process_proxy((args[0],), (orig_asarray,))[0]

    return orig_asarray(*args, **kwargs)


np.asarray = lazy_series_asarray


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

    out = alepython.ale.ale_plot(
        model=model,
        train_set=X_train,
        features=column,
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
    ale_factor_exp=0,
    x_factor_exp=0,
    x_ndigits=2,
    y_factor_exp=0,
    y_ndigits=2,
    *,
    experiment,
):
    model.n_jobs = n_jobs

    cbar_width = 0.01

    x_coords = {}
    x_coords["ALE start"] = 0
    x_coords["ALE end"] = 0.42
    x_coords["ALE cbar start"] = x_coords["ALE end"] + 0.01
    x_coords["ALE cbar end"] = x_coords["ALE cbar start"] + cbar_width
    x_coords["Samples start"] = 0.6
    x_coords["Samples end"] = 0.9
    x_coords["Samples cbar start"] = x_coords["Samples end"] + 0.01
    x_coords["Samples cbar end"] = x_coords["Samples cbar start"] + cbar_width

    y_bottom = {
        "Samples": 0.22,  # Samples plot and cbar bottom.
    }
    cbar_height = {
        "ALE": 0.6,
        "Samples": 0.35,
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
        fig, axes, (quantiles_list, ale, samples) = alepython.ale.ale_plot(
            model=model,
            train_set=train_set,
            features=features,
            bins=20,
            fig=fig,
            ax=ax[0],
            plot_quantiles=False,
            quantile_axis=True,
            plot_kwargs={
                "kind": "grid",
                "cmap": "inferno",
                "colorbar_kwargs": dict(
                    format=get_float_format(factor=10 ** ale_factor_exp, ndigits=0),
                    cax=cax[0],
                    label=(
                        f"ALE (BA)"
                        if ale_factor_exp == 0
                        else f"ALE ($10^{{{ale_factor_exp}}}$ BA)"
                    ),
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
    axes["ale"].set_title("")

    x_factor = 10 ** x_factor_exp
    y_factor = 10 ** y_factor_exp

    axes["ale"].xaxis.set_ticklabels(
        np.vectorize(get_float_format(factor=x_factor, ndigits=x_ndigits, atol=np.inf))(
            quantiles_list[0]
        )
    )
    axes["ale"].yaxis.set_ticklabels(
        np.vectorize(get_float_format(factor=y_factor, ndigits=y_ndigits, atol=np.inf))(
            quantiles_list[1]
        )
    )

    axes["ale"].set_xlabel(
        f"{features[0]} ({variable.units[features[0].parent]})"
        if x_factor_exp == 0
        else f"{features[0]} ($10^{{{x_factor_exp}}}$ {variable.units[features[0].parent]})"
    )
    axes["ale"].set_ylabel(
        f"{features[1]} ({variable.units[features[1].parent]})"
        if y_factor_exp == 0
        else f"{features[1]} ($10^{{{y_factor_exp}}}$ {variable.units[features[1].parent]})"
    )

    for tick in axes["ale"].xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment("right")

    if plot_samples:
        # Plotting samples.
        mod_quantiles_list = []
        for axis, quantiles, factor, ndigits in zip(
            ("x", "y"), quantiles_list, (x_factor, y_factor), (x_ndigits, y_ndigits)
        ):
            inds = np.arange(len(quantiles))
            mod_quantiles_list.append(inds)
            ax[1].set(**{f"{axis}ticks": inds})
            ax[1].set(
                **{
                    f"{axis}ticklabels": np.vectorize(
                        get_float_format(factor=factor, ndigits=ndigits, atol=np.inf)
                    )(quantiles)
                }
            )
            for label in getattr(ax[1], f"{axis}axis").get_ticklabels()[1::2]:
                label.set_visible(False)

        samples_img = ax[1].pcolormesh(
            *mod_quantiles_list, samples.T, norm=SymLogNorm(linthresh=1)
        )

        @ticker.FuncFormatter
        def integer_sci_format(x, pos):
            if np.log10(x).is_integer():
                return get_sci_format(ndigits=0, trim_leading_one=True)(x, pos)
            return ""

        fig.colorbar(
            samples_img,
            cax=cax[1],
            label="samples",
            format=integer_sci_format,
        )
        ax[1].xaxis.set_tick_params(rotation=50)
        for tick in ax[1].xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment("right")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel(
            f"{features[0]} ({variable.units[features[0].parent]})"
            if x_factor_exp == 0
            else f"{features[0]} ($10^{{{x_factor_exp}}}$ {variable.units[features[0].parent]})"
        )
        ax[1].set_ylabel(
            f"{features[1]} ({variable.units[features[1].parent]})"
            if y_factor_exp == 0
            else f"{features[1]} ($10^{{{y_factor_exp}}}$ {variable.units[features[1].parent]})"
        )

        fig.set_constrained_layout_pads(
            w_pad=0.000, h_pad=0.000, hspace=0.0, wspace=0.015
        )

    if figure_saver is not None:
        name_root = (
            experiment.name
            + "_"
            + "__".join(map(shorten_features, map(str, features))).replace(" ", "_")
        )
        if plot_samples:
            figure_saver.save_figure(
                fig,
                name_root,
                sub_directory="2d_ale_first_order" if include_first_order else "2d_ale",
            )
        else:
            figure_saver.save_figure(
                fig,
                name_root + "_no_count",
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
    ) = alepython.ale.multi_ale_plot_1d(
        model=model,
        train_set=X_train,
        bins=20,
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
            fig,
            "__".join(map(shorten_features(map(str, features)))).replace(" ", "_"),
            sub_directory=sub_dir,
        )

    return final_quantiles


def get_model_predict(model):
    return model.predict


def single_ax_multi_ale_1d(
    ax,
    feature_data,
    feature,
    xlabel=None,
    ylabel=None,
    title=None,
    verbose=False,
    x_ndigits=2,
    x_factor=1,
    x_rotation=18,
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
                bins=20,
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
                lambda x: get_float_format(
                    ndigits=x_ndigits, factor=x_factor, atol=np.inf
                )(x, None),
                final_quantiles[::2],
            )
        )
        ax.xaxis.set_tick_params(rotation=x_rotation)

        ax.grid(True)

        ax.set_xlabel(xlabel + f"({x_factor})")
        ax.set_ylabel(ylabel)

    ax.set_title(title)


def multi_model_ale_1d(
    variable_factory,
    experiment_data,
    experiment_plot_kwargs,
    lags=(0, 1, 3, 6, 9),
    title=None,
    verbose=False,
    figure_saver=None,
    single_figsize=(5.4, 1.5),
    legend_bbox=(0.5, 0.5),
    fig=None,
    axes=None,
    legend=True,
    x_ndigits=2,
    x_factor=1,
    x_rotation=18,
    y_ndigits=1,
    y_factor=1e-3,
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
            xlabel=str(feature),
            verbose=verbose,
            x_ndigits=x_ndigits,
            x_factor=x_factor,
            x_rotation=x_rotation,
        )

    for ax in axes.flatten()[:n_plots]:
        ax.yaxis.set_major_formatter(
            get_float_format(factor=y_factor, ndigits=y_ndigits, atol_exceeded="adjust")
        )

    for row_axes in axes:
        row_axes[0].set_ylabel(update_label_with_exp("ALE (BA)", str(y_factor)))

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
