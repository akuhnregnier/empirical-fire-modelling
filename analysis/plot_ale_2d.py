# -*- coding: utf-8 -*-
"""2D ALE plotting."""
import logging
import sys
import warnings
from itertools import combinations, islice
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger
from wildfires.qstat import get_ncpus

import empirical_fire_modelling.plotting.configuration as plotting_configuration
from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.ale import save_ale_2d
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


def plot_2d_ale(experiment, single=False, nargs=None, verbose=False, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    columns_list = list(combinations(X_train.columns, 2))

    # Deterministic sorting with FAPAR & FAPAR 1M and FAPAR & DRY_DAY_PERIOD at the
    # front since these are used in the paper.

    def get_combination_value(column_combination):
        # Handle special cases first.
        if (
            variable.FAPAR[0] in column_combination
            and variable.FAPAR[1] in column_combination
        ):
            return -1000
        elif (
            variable.FAPAR[0] in column_combination
            and variable.DRY_DAY_PERIOD[0] in column_combination
        ):
            return -999
        out = ""
        for var in column_combination:
            out += str(var.rank) + str(var.shift)
        return int(out)

    columns_list = sorted(columns_list, key=get_combination_value)

    def param_iter():
        for columns in columns_list:
            for plot_samples in [True, False]:
                yield columns, plot_samples

    if single:
        total = 1
    elif nargs:
        total = nargs
    else:
        total = 2 * len(columns_list)

    for columns, plot_samples in tqdm(
        islice(param_iter(), None, total),
        desc=f"2D ALE plotting ({experiment})",
        total=total,
        disable=not verbose,
    ):
        save_ale_2d(
            experiment=experiment,
            model=model,
            train_set=X_train,
            features=columns,
            n_jobs=get_ncpus(),
            include_first_order=True,
            plot_samples=plot_samples,
            figure_saver=exp_figure_saver,
            ale_factor_exp=plotting_configuration.ale_factor_exps.get(
                (columns[0].parent, columns[1].parent), -2
            ),
            x_factor_exp=plotting_configuration.factor_exps.get(columns[0].parent, 0),
            x_ndigits=plotting_configuration.ndigits.get(columns[0].parent, 2),
            y_factor_exp=plotting_configuration.factor_exps.get(columns[1].parent, 0),
            y_ndigits=plotting_configuration.ndigits.get(columns[1].parent, 2),
        )
        plt.close("all")


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="10GB")

    run(plot_2d_ale, list(Experiment), cx1_kwargs=cx1_kwargs)
