# -*- coding: utf-8 -*-
"""Combination of vegetation and dry-day period 1D ALE plots."""
import logging
import sys
import warnings
from operator import itemgetter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from loguru import logger as loguru_logger
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from wildfires.utils import shorten_features

import empirical_fire_modelling.plotting.configuration as plotting_configuration
from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.ale import multi_ale_1d
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import (
    get_experiment_split_data,
    get_frac_train_nr_samples,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.plotting import figure_saver
from empirical_fire_modelling.utils import tqdm

mpl.rc_file(Path(__file__).resolve().parent / "matplotlibrc")

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging(level="WARNING")

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)


def plot_multi_ale(experiment, verbose=False, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8))

    expected_veg = tuple(
        map(
            itemgetter(0),
            variable.feature_categories[variable.Category.VEGETATION],
        )
    )

    matched = [f for f in expected_veg if f in X_train.columns]

    if len(matched) == 0:
        raise ValueError(f"Could not find one of {expected_veg} in {X_train.columns}.")
    elif len(matched) > 1:
        raise ValueError(
            f"Found more than one of {tuple(map(str, expected_veg))} in "
            f"{X_train.columns}: {matched}"
        )
    features = (matched[0].parent, variable.DRY_DAY_PERIOD)

    ale_factor_exp = -3
    x_factor_exp = 0

    for feature_factory, ax, title in zip(
        tqdm(features, desc="Processing features"),
        axes,
        ("(a)", "(b)"),
    ):
        multi_ale_1d(
            model=model,
            X_train=X_train,
            features=[feature_factory[lag] for lag in variable.lags[:5]],
            train_response=y_train,
            fig=fig,
            ax=ax,
            verbose=verbose,
            monte_carlo_rep=100,
            monte_carlo_ratio=get_frac_train_nr_samples(Experiment["15VEG_FAPAR"], 0.1),
            legend=False,
            ale_factor_exp=ale_factor_exp,
            x_factor_exp=x_factor_exp,
            x_ndigits=plotting_configuration.ndigits.get(feature_factory, 2),
            x_skip=4,
            x_rotation=0,
        )
        ax.set_title(title)
        ax.set_xlabel(
            f"{shorten_features(str(feature_factory))} ({variable.units[feature_factory]})"
            if x_factor_exp == 0
            else (
                f"{feature_factory} ($10^{{{x_factor_exp}}}$ "
                f"{variable.units[feature_factory]})"
            ),
        )

    axes[1].set_ylabel("")

    # Inset axis to pronounce low-DD features.

    ax2 = inset_axes(
        axes[1],
        width=2.155,
        height=1.55,
        loc="lower left",
        bbox_to_anchor=(0.019, 0.225),
        bbox_transform=ax.transAxes,
    )
    # Plot the DD data again on the inset axis.
    multi_ale_1d(
        model=model,
        X_train=X_train,
        features=[features[1][lag] for lag in variable.lags[:5]],
        train_response=y_train,
        fig=fig,
        ax=ax2,
        verbose=verbose,
        monte_carlo_rep=100,
        monte_carlo_ratio=get_frac_train_nr_samples(Experiment["15VEG_FAPAR"], 0.1),
        legend=False,
        ale_factor_exp=ale_factor_exp,
    )

    ax2.set_xlim(0, 17.5)
    ax2.set_ylim(-1.5e-3, 2e-3)

    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.tick_params(axis="both", which="both", length=0)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax2.grid(True)

    mark_inset(axes[1], ax2, loc1=4, loc2=2, fc="none", ec="0.3")

    # Move the first (left) axis to the right.
    orig_bbox = axes[0].get_position()
    axes[0].set_position(
        [orig_bbox.xmin + 0.021, orig_bbox.ymin, orig_bbox.width, orig_bbox.height]
    )

    # Explicitly set the x-axis labels' positions so they line up horizontally.
    y_min = 1
    for ax in axes:
        bbox = ax.get_position()
        if bbox.ymin < y_min:
            y_min = bbox.ymin
    for ax in axes:
        bbox = ax.get_position()
        mean_x = (bbox.xmin + bbox.xmax) / 2.0
        # NOTE - Decrease the negative offset to move the label upwards.
        ax.xaxis.set_label_coords(mean_x, y_min - 0.1, transform=fig.transFigure)

    # Plot the legend in between the two axes.
    axes[1].legend(
        loc="center",
        ncol=5,
        bbox_to_anchor=(
            np.mean(
                [
                    axes[0].get_position().xmax,
                    axes[1].get_position().xmin,
                ]
            ),
            0.932,
        ),
        bbox_transform=fig.transFigure,
        handletextpad=0.25,
        columnspacing=0.5,
    )

    exp_figure_saver.save_figure(
        fig,
        f'{experiment.name}_{"__".join(map(shorten_features, map(str, features)))}_ale_shifts',
        sub_directory="multi_ale",
        transparent=False,
    )


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="24:00:00", ncpus=32, mem="60GB")
    run(plot_multi_ale, list(Experiment), cx1_kwargs=cx1_kwargs)
