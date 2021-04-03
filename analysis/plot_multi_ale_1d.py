# -*- coding: utf-8 -*-
"""Combination of vegetation and dry-day period 1D ALE plots."""
import logging
import sys
import warnings
from operator import itemgetter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as loguru_logger

from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.ale import multi_ale_1d
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
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

    for feature_factory, ax, title in zip(
        tqdm(features, desc="Processing features"), axes, ("(a)", "(b)")
    ):
        final_quantiles = multi_ale_1d(
            model=model,
            X_train=X_train,
            features=[feature_factory[lag] for lag in variable.lags[:5]],
            train_response=y_train,
            fig=fig,
            ax=ax,
            verbose=verbose,
            monte_carlo_rep=100,
            monte_carlo_ratio=0.1,
        )
        ax.set_title(title)

        min_abs = np.min(np.abs(final_quantiles))
        if 0 < min_abs < 1:
            precision = abs(round(np.floor(np.log10(min_abs))))
        precision = min(max(precision, 0), 5)  # Clip the precision.
        ax.set_xticklabels(
            tuple(map(lambda s: format(s, f"0.{precision}f"), final_quantiles))
        )

        ax.xaxis.set_tick_params(rotation=45)

    axes[0].set_ylabel("ALE (BA)")

    fig.tight_layout(w_pad=0.02)
    fig.align_labels()

    # Explicitly set the x-axis labels' positions so they line up horizontally.
    y_min = 1
    for ax in axes:
        bbox = ax.get_position()
        if bbox.ymin < y_min:
            y_min = bbox.ymin
    for ax in axes:
        bbox = ax.get_position()
        mean_x = (bbox.xmin + bbox.xmax) / 2.0
        ax.xaxis.set_label_coords(mean_x, y_min - 0.147, transform=fig.transFigure)

    exp_figure_saver.save_figure(
        fig,
        f'{"__".join(map(str, features))}_ale_shifts',
        sub_directory="multi_ale",
    )


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="8GB")
    run(plot_multi_ale, list(Experiment), cx1_kwargs=cx1_kwargs)
