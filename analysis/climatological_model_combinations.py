# -*- coding: utf-8 -*-
"""Fitting of various model combinations."""
import logging
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.model_combinations import fit_combination
from empirical_fire_modelling.configuration import Experiment, n_splits
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging

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


def combination_fit(combination, split_index, cache_check=False, **kwargs):
    # Get training and test data for all variables.
    get_experiment_split_data.check_in_store(Experiment.ALL)
    X_all, _, y, _ = get_experiment_split_data(Experiment.ALL)

    if cache_check:
        return fit_combination.check_in_store(X_all, y, combination, split_index)
    return fit_combination(X_all, y, combination, split_index)


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="24:00:00", ncpus=1, mem="7GB")

    # Get training and test data for all variables.
    get_experiment_split_data.check_in_store(Experiment.ALL)
    X_train, X_test, y_train, y_test = get_experiment_split_data(Experiment.ALL)

    shifts = (0, 1, 3, 6, 9)
    assert all(shift in variable.lags for shift in shifts)

    veg_lags = tuple(
        tuple(
            [
                var_factory[shift]
                for var_factory in variable.feature_categories[
                    variable.Category.VEGETATION
                ]
            ]
        )
        for shift in shifts
    )

    assert all(feature in X_train for unpacked in veg_lags for feature in unpacked)
    assert all(feature in X_test for unpacked in veg_lags for feature in unpacked)

    combinations = [
        (
            variable.DRY_DAY_PERIOD[0],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[9],
            variable.POPD[0],
            variable.DRY_DAY_PERIOD[6],
            variable.LIGHTNING[0],
            variable.DIURNAL_TEMP_RANGE[0],
            *veg_lag_product,
        )
        for veg_lag_product in product(*veg_lags)
    ]

    assert all(len(combination) == 15 for combination in combinations)

    args = [[], []]

    for combination in combinations:
        for i in range(n_splits):
            args[0].append(combination)
            args[1].append(i)

    args_scores = run(
        combination_fit, *args, cx1_kwargs=cx1_kwargs, return_local_args=True
    )

    if args_scores is None or (
        isinstance(args_scores, dict)
        and set(args_scores)
        == {
            "present",
            "uncached",
        }
    ):
        sys.exit(0)
