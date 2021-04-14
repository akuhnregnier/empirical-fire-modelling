# -*- coding: utf-8 -*-
"""BA plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger
from wildfires.utils import get_land_mask

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import (
    ba_dataset_map,
    get_data,
    get_experiment_split_data,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import ba_plotting, map_figure_saver
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
    exp_figure_saver = map_figure_saver(sub_directory=experiment.name)

    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached data only.
    get_data(experiment, cache_check=True)
    master_mask = get_data(experiment)[2]

    single_master_mask = master_mask[0]

    if not all(
        np.all(master_mask[i] == single_master_mask)
        for i in range(1, master_mask.shape[0])
    ):
        raise ValueError("master_mask should be the same across all times.")

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)

    ba_data = ba_dataset_map[y_test.name]().get_mean_dataset().cube.data

    land_mask = get_land_mask()

    # Indicate areas with 0 BA but with BA data availability (but without data
    # availability otherwise).
    unique_ba_values = np.unique(ba_data)
    zero_ba = (ba_data.data < unique_ba_values[1]) & land_mask & single_master_mask

    # Indicate areas with nonzero BA but with BA data availability (but without data
    # availability otherwise).
    nonzero_ba = (
        (ba_data.data.data > unique_ba_values[0]) & land_mask & single_master_mask
    )

    predicted_test = threading_get_model_predict(
        X_train=X_train,
        y_train=y_train,
        predict_X=X_test,
    )

    ba_plotting(
        get_mm_data(predicted_test, master_mask, kind="val"),
        get_mm_data(y_test.values, master_mask, kind="val"),
        figure_saver=exp_figure_saver,
        aux0=zero_ba,
        aux0_label="BA = 0",
        aux1=nonzero_ba,
        aux1_label="BA > 0",
    )


if __name__ == "__main__":
    run(plot_ba, list(Experiment), cx1_kwargs=False)
