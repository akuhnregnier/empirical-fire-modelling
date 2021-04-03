# -*- coding: utf-8 -*-
"""1D ALE plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.analysis.ale import save_ale_1d
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


def plot_1d_ale(
    experiment, plot_monte_carlo=True, single=False, verbose=False, **kwargs
):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    params = []
    for column in X_train.columns:
        params.append(column)

    if single:
        total = 1
    else:
        total = len(params)

    for column in tqdm(
        params[:total],
        desc=f"1D ALE plotting ({experiment})",
        disable=not verbose,
    ):
        save_ale_1d(
            model,
            X_train,
            column,
            train_response=y_train,
            figure_saver=exp_figure_saver,
            verbose=verbose,
            monte_carlo_rep=200,
            monte_carlo_ratio=0.1,
        )


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="8GB")
    run(plot_1d_ale, list(Experiment), plot_monte_carlo=False, cx1_kwargs=cx1_kwargs)
