# -*- coding: utf-8 -*-
"""1D ALE plotting."""
import logging
import sys
import warnings
from operator import add, attrgetter, sub
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as loguru_logger
from matplotlib.lines import Line2D

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, get_model_scores
from empirical_fire_modelling.plotting import figure_saver

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


def plot_score_groups(experiments, **kwargs):
    scores = {}
    for experiment in experiments:
        # Operate on cached data only.
        get_experiment_split_data.check_in_store(experiment)
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

        # Operate on cached fitted models only.
        get_model(X_train, y_train, cache_check=True)
        model = get_model(X_train, y_train)

        # Cached scores only.
        get_model_scores.check_in_store(model, X_test, X_train, y_test, y_train)
        scores[experiment] = get_model_scores(model, X_test, X_train, y_test, y_train)

    # Sort scores based on the validation R2 score.
    sort_indices = np.argsort([score["test_r2"] for score in scores.values()])[::-1]

    # Sorted values.
    s_train_r2s = np.array([score["train_r2"] for score in scores.values()])[
        sort_indices
    ]
    s_validation_r2s = np.array([score["test_r2"] for score in scores.values()])[
        sort_indices
    ]
    s_oob_r2s = np.array([score["oob_r2"] for score in scores.values()])[sort_indices]

    # Adapted from: https://matplotlib.org/gallery/subplots_axes_and_figures/broken_axis.html

    # Ratio of training R2 range to validation R2 range.
    train_validation_ratio = np.ptp(s_train_r2s) / np.ptp(s_validation_r2s)

    fig = plt.figure(figsize=(4, 2.2), dpi=200)

    all_ax = fig.add_subplot(1, 1, 1)
    all_ax.set_ylabel(r"$\mathrm{R}^2$", labelpad=29)
    all_ax.set_xticks([])
    all_ax.set_yticks([])
    all_ax.set_frame_on(
        False
    )  # So we don't get black bars showing through the 'broken' gap.

    # Break the y-axis into 2 parts.
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3.5))
    ax1, ax2 = fig.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[train_validation_ratio, 1])
    )
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # Plot train and validation R2s.

    train_kwargs = dict(linestyle="", marker="x", c="C1", label="train")
    ax1.plot(s_train_r2s, **train_kwargs)

    validation_kwargs = dict(linestyle="", marker="o", c="C0", label="validation")
    ax2.plot(s_validation_r2s, **validation_kwargs)

    oob_kwargs = dict(linestyle="", marker="^", c="C2", label="train OOB")
    ax2.plot(s_oob_r2s, **oob_kwargs)

    ax2.set_yticks(np.arange(0.575, 0.7 + 0.01, 0.025))

    ax2.legend(
        handles=[
            Line2D([0], [0], **kwargs)
            for kwargs in (train_kwargs, validation_kwargs, oob_kwargs)
        ],
        loc="lower left",
    )

    ylim_1 = ax1.get_ylim()
    ylim_2 = ax2.get_ylim()

    margin_f = (0.22, 0.05)  # Two-sided relative margin addition.
    ax1.set_ylim(
        [
            op(ylim_val, factor * np.ptp(ylim_1))
            for ylim_val, factor, op in zip(ylim_1, margin_f, (sub, add))
        ]
    )
    ax2.set_ylim(
        [
            op(ylim_val, factor * np.ptp(ylim_1) / train_validation_ratio)
            for ylim_val, factor, op in zip(ylim_2, margin_f, (sub, add))
        ]
    )
    # ax2.set_ylim(ylim_2[0], ylim_2[1] + margin_f * np.ptp(ylim_1) / train_validation_ratio)

    # hide the spines between ax and ax2
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1.xaxis.set_ticks_position("none")  # hide top ticks themselves (not just labels)

    ax2.xaxis.tick_bottom()

    ax2.set_xticks(list(range(len(experiments))))
    ax2.set_xticklabels(
        list(np.array(list(map(attrgetter("name"), scores)))[sort_indices]),
        rotation=45,
        ha="right",
    )
    ax2.tick_params(axis="x", which="major", pad=0)

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    for ax in (ax1, ax2):
        ax.set_xticks(list(range(len(experiments))))

    figure_saver.save_figure(fig, "model_comp_scores")


if __name__ == "__main__":
    experiment_groups = (
        (
            Experiment.ALL,
            Experiment.TOP15,
            Experiment.CURR,
            Experiment["15VEG_FAPAR"],
            Experiment["15VEG_LAI"],
            Experiment["15VEG_SIF"],
            Experiment["15VEG_VOD"],
            Experiment.CURRDD_FAPAR,
            Experiment.CURRDD_LAI,
            Experiment.CURRDD_SIF,
            Experiment.CURRDD_VOD,
            Experiment.BEST15,
        ),
    )
    run(plot_score_groups, experiment_groups, cx1_kwargs=False)
