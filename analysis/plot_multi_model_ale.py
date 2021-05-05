# -*- coding: utf-8 -*-
"""BA prediction comparison plots."""
import logging
import re
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger
from wildfires.utils import shorten_features

from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.ale import multi_model_ale_1d
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import (
    get_data,
    get_endog_exog_mask,
    get_experiment_split_data,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.plotting import experiment_plot_kwargs, figure_saver
from empirical_fire_modelling.utils import check_master_masks, tqdm

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


def multi_model_ale_plot(*args, verbose=False, **kwargs):
    # Experiments for which data will be plotted.
    experiments = [
        Experiment["ALL"],
        Experiment["TOP15"],
        Experiment["CURR"],
        Experiment["BEST15"],
        Experiment["15VEG_FAPAR"],
        Experiment["15VEG_LAI"],
        Experiment["15VEG_VOD"],
        Experiment["15VEG_SIF"],
        Experiment["CURRDD_FAPAR"],
        Experiment["CURRDD_LAI"],
        Experiment["CURRDD_VOD"],
        Experiment["CURRDD_SIF"],
    ]

    # Operate on cached data/models only.
    experiment_masks = []
    plotting_experiment_data = {}

    for experiment in tqdm(experiments, desc="Loading data"):
        get_data(experiment, cache_check=True)
        get_experiment_split_data.check_in_store(experiment)
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)
        get_model(X_train, y_train, cache_check=True)

        experiment_masks.append(get_endog_exog_mask(experiment)[2])
        plotting_experiment_data[experiment] = dict(
            model=get_model(X_train, y_train),
            X_train=X_train,
        )

    # Ensure masks are aligned.
    check_master_masks(*experiment_masks)

    lags = (0, 1, 3, 6, 9)

    for comp_vars in [[variable.FAPAR, variable.LAI], [variable.SIF, variable.VOD]]:
        fig, axes = plt.subplots(5, 2, sharex="col", figsize=(7.0, 5.8))

        # Create general legend labels (with 'X' instead of FAPAR, or LAI, etc...).
        mod_exp_plot_kwargs = deepcopy(experiment_plot_kwargs)
        for plot_kwargs in mod_exp_plot_kwargs.values():
            if plot_kwargs["label"].startswith("15VEG_"):
                plot_kwargs["label"] = "15VEG_X"
            elif plot_kwargs["label"].startswith("CURRDD_"):
                plot_kwargs["label"] = "CURRDD_X"

        x_factor_exp = 0
        x_factor = 10 ** x_factor_exp
        # x_factor_str = rf"$10^{{{x_factor_exp}}}$"

        y_factor_exp = -4
        y_factor = 10 ** y_factor_exp
        y_factor_str = rf"$10^{{{y_factor_exp}}}$"

        multi_model_ale_1d(
            comp_vars[0],
            plotting_experiment_data,
            mod_exp_plot_kwargs,
            verbose=verbose,
            legend_bbox=(0.5, 1.01),
            fig=fig,
            axes=axes[:, 0:1],
            lags=lags,
            x_ndigits=2,
            x_factor=x_factor,
            x_rotation=0,
            y_ndigits=0,
            y_factor=y_factor,
        )
        multi_model_ale_1d(
            comp_vars[1],
            plotting_experiment_data,
            experiment_plot_kwargs,
            verbose=verbose,
            legend=False,
            fig=fig,
            axes=axes[:, 1:2],
            lags=lags,
            x_ndigits=2,
            x_factor=x_factor,
            x_rotation=0,
            y_ndigits=0,
            y_factor=y_factor,
        )

        for ax in axes[:, 1]:
            ax.set_ylabel("")
        for ax in axes[:, 0]:
            lag_match = re.search("(\dM)", ax.get_xlabel())
            if lag_match:
                lag_m = f" {lag_match.group(1)}"
            else:
                lag_m = ""
            ax.set_ylabel(f"ALE{lag_m} ({y_factor_str} BA)")
        for ax in axes.flatten():
            ax.set_xlabel("")

        for ax, var in zip(axes[-1], comp_vars):
            assert x_factor_exp == 0
            ax.set_xlabel(f"{shorten_features(str(var))} ({variable.units[var]})")

        for ax, title in zip(axes.flatten(), ascii_lowercase):
            ax.text(0.5, 1.05, f"({title})", transform=ax.transAxes)

        fig.tight_layout(h_pad=0.4)
        fig.align_labels()

        figure_saver.save_figure(
            fig,
            f"{'__'.join(map(shorten_features, map(str, comp_vars)))}_ale_comp",
            sub_directory="ale_comp",
        )


if __name__ == "__main__":
    run(multi_model_ale_plot, [None], cx1_kwargs=False)
