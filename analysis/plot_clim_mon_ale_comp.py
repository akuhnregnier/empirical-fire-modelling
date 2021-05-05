# -*- coding: utf-8 -*-
"""Comparison of 1D ALE plots for climatological and monthly analyses."""
import gc
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger

import empirical_fire_modelling.plotting.configuration as plotting_configuration
from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.ale import save_ale_1d
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


def skip_14(labels):
    """Customised label skipping for 14 quantiles."""
    return (
        labels[0],
        labels[2],
        labels[4],
        labels[6],
        labels[8],
        labels[10],
        labels[13],
    )


def plot_single_1d_ale(experiment, column, ax, verbose=False):
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    save_ale_1d(
        model,
        X_train,
        column,
        train_response=y_train,
        figure_saver=None,
        verbose=verbose,
        monte_carlo_rep=100,
        monte_carlo_ratio=get_frac_train_nr_samples(Experiment["15VEG_FAPAR"], 0.1),
        ax=ax,
        ale_factor_exp=plotting_configuration.ale_factor_exps[column.parent],
        x_ndigits=plotting_configuration.ndigits.get(column.parent, 2),
        x_skip=4
        if (
            (experiment, column)
            != (Experiment["15VEG_FAPAR_MON"], variable.DRY_DAY_PERIOD[3])
        )
        else skip_14,
    )


def plot_clim_mon_ale_comp(*args, verbose=False, **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.1))
    plot_spec = {
        axes[0, 0]: (Experiment["15VEG_FAPAR"], variable.FAPAR[0], "(a) 15VEG_FAPAR"),
        axes[0, 1]: (
            Experiment["15VEG_FAPAR_MON"],
            variable.FAPAR[0],
            "(b) 15VEG_FAPAR_MON",
        ),
        axes[1, 0]: (
            Experiment["15VEG_FAPAR"],
            variable.DRY_DAY_PERIOD[3],
            "(c) 15VEG_FAPAR",
        ),
        axes[1, 1]: (
            Experiment["15VEG_FAPAR_MON"],
            variable.DRY_DAY_PERIOD[3],
            "(d) 15VEG_FAPAR_MON",
        ),
    }
    for (ax, (experiment, column, title)) in tqdm(
        plot_spec.items(), desc="ALE plots", disable=not verbose
    ):
        plot_single_1d_ale(experiment, column, ax=ax, verbose=verbose)
        ax.set_title(title)
        gc.collect()

    for ax in axes[:, 1]:
        ax.set_ylabel("")

    fig.tight_layout()
    fig.align_labels()

    figure_saver.save_figure(fig, "15VEG_FAPAR_15VEG_FAPAR_MON_ALE_comp")


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="24:00:00", ncpus=32, mem="60GB")
    run(plot_clim_mon_ale_comp, [None], cx1_kwargs=cx1_kwargs)
