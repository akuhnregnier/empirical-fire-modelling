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
from empirical_fire_modelling.data import get_endog_exog_mask, get_experiment_split_data
from empirical_fire_modelling.data.cached_processing import get_ba_plotting_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import (
    ba_plotting,
    get_aux0_aux1_kwargs,
    map_figure_saver,
)
from empirical_fire_modelling.utils import check_master_masks

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
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached data only.
    get_endog_exog_mask.check_in_store(experiment)
    master_mask = get_endog_exog_mask(experiment)[2]

    check_master_masks(master_mask)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)

    predicted_test = threading_get_model_predict(
        X_train=X_train,
        y_train=y_train,
        predict_X=X_test,
    )

    ba_plotting(
        *get_ba_plotting_data(predicted_test, y_test, master_mask),
        figure_saver=map_figure_saver(sub_directory=experiment.name),
        **get_aux0_aux1_kwargs(y_test, master_mask),
        filename=f"{experiment.name}_ba_prediction",
    )


if __name__ == "__main__":
    run(plot_ba, list(Experiment), cx1_kwargs=False)
