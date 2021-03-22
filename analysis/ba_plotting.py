# -*- coding: utf-8 -*-
"""BA plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_data, get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.plotting import ba_plotting, figure_saver
from empirical_fire_modelling.utils import get_mm_data

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


def plot_ba(experiment, **kwargs):
    exp_figure_saver = figure_saver(sub_directory=str(experiment))

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached data only.
    get_data(experiment, cache_check=True)
    master_mask = get_data(experiment)[2]

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    model = get_model(X_train, y_train)

    predicted_train = model.predict(X_train)

    ba_plotting(
        get_mm_data(predicted_train, master_mask, kind="train"),
        get_mm_data(y_train.values, master_mask, kind="train"),
        figure_saver=exp_figure_saver,
    )


if __name__ == "__main__":
    run(plot_ba, list(Experiment), cx1_kwargs=False)
