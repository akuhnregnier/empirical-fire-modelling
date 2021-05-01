# -*- coding: utf-8 -*-
"""BA prediction comparison plots."""
import logging
import sys
import warnings
from operator import attrgetter
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger
from wildfires.data import dummy_lat_lon_cube
from wildfires.utils import get_unmasked

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.data import (
    get_data,
    get_endog_exog_mask,
    get_experiment_split_data,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import (
    disc_cube_plot,
    get_aux0_aux1_kwargs,
    map_figure_saver,
)
from empirical_fire_modelling.utils import check_master_masks, get_mm_data

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


def prediction_comparisons():
    """Compare ALL and CURR predictions."""
    experiments = [Experiment.ALL, Experiment.CURR]
    # Operate on cached data/models only.

    experiment_data = {}
    experiment_models = {}

    for experiment in experiments:
        get_data(experiment, cache_check=True)
        get_experiment_split_data.check_in_store(experiment)
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)
        get_model(X_train, y_train, cache_check=True)

        experiment_data[experiment] = get_endog_exog_mask(experiment)
        experiment_models[experiment] = get_model(X_train, y_train)

    # Ensure masks are aligned.
    check_master_masks(*(data[2] for data in experiment_data.values()))

    master_mask = next(iter(experiment_data.values()))[2]

    # Record predictions and errors.
    experiment_predictions = {}
    experiment_errors = {}
    map_experiment_predictions = {}
    map_experiment_errors = {}

    for experiment in experiments:
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)
        predicted_test = threading_get_model_predict(
            X_train=X_train,
            y_train=y_train,
            predict_X=X_test,
        )
        experiment_predictions[experiment] = predicted_test
        experiment_errors[experiment] = y_test.values - predicted_test

        map_experiment_predictions[experiment] = get_mm_data(
            experiment_predictions[experiment], master_mask, kind="val"
        )
        map_experiment_errors[experiment] = get_mm_data(
            experiment_errors[experiment], master_mask, kind="val"
        )

    error_mag_diff = np.abs(map_experiment_errors[experiments[1]]) - np.abs(
        map_experiment_errors[experiments[0]]
    )

    y_test = get_experiment_split_data(experiment)[3]

    rel_error_mag_diff = np.mean(error_mag_diff, axis=0) / np.mean(
        get_mm_data(y_test.values, master_mask, kind="val"), axis=0
    )
    all_rel = get_unmasked(rel_error_mag_diff)

    print(f"% >0: {100 * np.sum(all_rel > 0) / all_rel.size:0.1f}")
    print(f"% <0: {100 * np.sum(all_rel < 0) / all_rel.size:0.1f}")

    fig, ax, cbar = disc_cube_plot(
        dummy_lat_lon_cube(rel_error_mag_diff),
        bin_edges=(-0.5, 0, 0.5),
        extend="both",
        cmap="PiYG",
        cmap_midpoint=0,
        cmap_symmetric=False,
        cbar_label=f"<|Err({experiments[1].name})| - |Err({experiments[0].name})|> / <Ob.>",
        cbar_shrink=0.3,
        cbar_aspect=15,
        cbar_extendfrac=0.1,
        cbar_pad=0.02,
        cbar_format=None,
        loc=(0.77, 0.15),
        **get_aux0_aux1_kwargs(y_test, master_mask),
    )
    cbar.ax.yaxis.label.set_size(7)

    map_figure_saver.save_figure(
        fig, f"rel_error_mag_diff_{'_'.join(map(attrgetter('name'), experiments))}"
    )


if __name__ == "__main__":
    prediction_comparisons()
