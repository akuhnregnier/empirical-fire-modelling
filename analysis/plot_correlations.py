# -*- coding: utf-8 -*-
"""Plotting of variable correlations."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger
from wildfires.analysis import corr_plot
from wildfires.utils import shorten_features

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_data, get_endog_exog_mask
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.plotting import figure_saver
from empirical_fire_modelling.variable import sort_variables

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


def correlation_plot(experiment, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_data(experiment, cache_check=True)
    _, exog_data, _ = get_endog_exog_mask(experiment)

    def df_cols_to_str(df):
        df.columns = list(map(lambda s: shorten_features(str(s)), df.columns))
        return df

    # with exp_figure_saver("corr_plot"):
    #     corr_plot(
    #         df_cols_to_str(
    #             exog_data[
    #                 list(
    #                     sort_variables(
    #                         var for var in exog_data.columns if var.shift <= 9
    #                     )
    #                 )
    #             ]
    #         ),
    #         fig_kwargs={"figsize": (12, 8)},
    #     )
    #     plt.grid(False)

    with exp_figure_saver(f"{experiment.name}_corr_plot_full"):
        corr_plot(
            df_cols_to_str(exog_data[list(sort_variables(exog_data.columns))]),
            rotation=70,
            fig_kwargs={"figsize": (8.2, 6.3)},
        )
        plt.grid(False)


if __name__ == "__main__":
    run(correlation_plot, list(Experiment), cx1_kwargs=False)
