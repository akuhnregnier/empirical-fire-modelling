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


def plot_2d_ale(experiment, single=False, verbose=False, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    columns_list = list(combinations(X_train.columns, 2))

    def param_iter():
        for columns in columns_list:
            for plot_samples in [True, False]:
                yield columns, plot_samples

    if single:
        total = 1
    else:
        total = 2 * len(columns_list)

    for columns, plot_samples in tqdm(
        islice(param_iter(), None, total),
        desc=f"2D ALE plotting ({experiment})",
        total=total,
        disable=not verbose,
    ):
        save_ale_2d(
            model=model,
            train_set=X_train,
            features=columns,
            n_jobs=get_ncpus(),
            include_first_order=True,
            plot_samples=plot_samples,
            figure_saver=exp_figure_saver,
        )
        plt.close("all")


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="10GB")

    run(plot_2d_ale, list(Experiment), cx1_kwargs=cx1_kwargs)
