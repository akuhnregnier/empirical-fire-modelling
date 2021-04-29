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
from empirical_fire_modelling.cx1 import get_parsers, run
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


def plot_1d_ale(experiment, column, single=False, verbose=False, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

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
        figure_saver=exp_figure_saver,
        verbose=verbose,
        monte_carlo_rep=200,
        monte_carlo_ratio=0.1,
    )


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="24:00:00", ncpus=32, mem="60GB")

    # Prepare arguments (experiment, column).
    args = [[], []]
    experiments = list(Experiment)

    cmd_args = get_parsers()["parser"].parse_args()

    if cmd_args.experiment is not None:
        chosen_experiments = [
            exp
            for exp in experiments
            if exp in tuple(Experiment[exp] for exp in cmd_args.experiment)
        ]
    else:
        chosen_experiments = experiments.copy()

    chosen_experiments = chosen_experiments[: 1 if cmd_args.single else None]

    for experiment in tqdm(
        chosen_experiments,
        desc="Preparing ALE 1D arguments",
        disable=not cmd_args.verbose,
    ):
        # Operate on cached data / models only.
        get_experiment_split_data.check_in_store(experiment)
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

        get_model(X_train, y_train, cache_check=True)

        for column in X_train.columns:
            args[0].append(experiment)
            args[1].extend(column)

    run(plot_1d_ale, *args, cx1_kwargs=cx1_kwargs)
