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
from wildfires.utils import get_land_mask, get_unmasked

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.data import (
    ba_dataset_map,
    get_data,
    get_experiment_split_data,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import disc_cube_plot, map_figure_saver
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

        experiment_data[experiment] = get_data(experiment)
        experiment_models[experiment] = get_model(X_train, y_train)

    master_mask = next(iter(experiment_data.values()))[2]
    single_master_mask = master_mask[0]

    # Ensure masks are aligned.
    for exp_master_mask in [data[2] for data in experiment_data.values()]:
        if not all(
            np.all(exp_master_mask[i] == single_master_mask)
            for i in range(1, exp_master_mask.shape[0])
        ):
            raise ValueError(
                "master_mask should be the same across all times and experiments."
            )

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
        aux0=zero_ba,
        aux0_label="BA = 0",
        aux1=nonzero_ba,
        aux1_label="BA > 0",
    )
    cbar.ax.yaxis.label.set_size(7)

    map_figure_saver.save_figure(
        fig, f"rel_error_mag_diff_{'_'.join(map(attrgetter('name'), experiments))}"
    )


if __name__ == "__main__":
    prediction_comparisons()
