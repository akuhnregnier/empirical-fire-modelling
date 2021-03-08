# -*- coding: utf-8 -*-
"""1D ALE plotting."""
import logging
import sys
import warnings
from itertools import islice
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger
from wildfires.qstat import get_ncpus

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
enable_logging()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)


def plot_1d_ale(experiment, single=False, verbose=False, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=str(experiment))

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    def param_iter():
        for column in X_train.columns:
            for monte_carlo in [False, True]:
                yield column, monte_carlo

    if single:
        total = 1
    else:
        total = X_train.shape[1] * 2

    for column, monte_carlo in tqdm(
        islice(param_iter(), None, total),
        desc=f"1D ALE plotting ({experiment})",
        total=total,
        disable=not verbose,
    ):
        save_ale_1d(
            model,
            X_train,
            column,
            n_jobs=get_ncpus(),
            monte_carlo=monte_carlo,
            figure_saver=exp_figure_saver,
        )


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="25GB")

    run(plot_1d_ale, list(Experiment), cx1_kwargs=cx1_kwargs)
