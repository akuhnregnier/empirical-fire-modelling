# -*- coding: utf-8 -*-
"""Fitting of various model combinations.

This should be run locally via a Dask client connected to several distributed workers.

"""
import logging
import sys
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as loguru_logger
from wildfires.dask_cx1 import DaskRandomForestRegressor, dask_fit_combinations
from wildfires.qstat import get_ncpus

from empirical_fire_modelling import variable
from empirical_fire_modelling.configuration import (
    CACHE_DIR,
    Experiment,
    n_splits,
    param_dict,
)
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.utils import get_client

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

if __name__ == "__main__":
    client = get_client(fallback=False)

    # Get training and test data for all variables.
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

    print("Starting fitting")

    scores = dask_fit_combinations(
        DaskRandomForestRegressor(**param_dict),
        X_train,
        y_train,
        client,
        combinations,
        n_splits=n_splits,
        local_n_jobs=max(get_ncpus() - 1, 1),
        verbose=True,
        cache_dir=CACHE_DIR,
    )

    r2_test_scores = {
        key: [data["test_score"][i]["r2"] for i in data["test_score"]]
        for key, data in scores.items()
    }
    mse_test_scores = {
        key: [data["test_score"][i]["mse"] for i in data["test_score"]]
        for key, data in scores.items()
    }

    keys = np.array(list(r2_test_scores))
    mean_r2_test_scores = np.array(
        [np.mean(scores) for scores in r2_test_scores.values()]
    )
    mean_mse_test_scores = np.array(
        [np.mean(scores) for scores in mse_test_scores.values()]
    )

    sort_indices = np.argsort(mean_r2_test_scores)[::-1]
    keys = keys[sort_indices]
    mean_r2_test_scores = mean_r2_test_scores[sort_indices]
    mean_mse_test_scores = mean_mse_test_scores[sort_indices]

    fig, ax = plt.subplots()
    ax.plot(mean_r2_test_scores)
    ax2 = ax.twinx()
    _ = ax2.plot(mean_mse_test_scores, c="C1")

    N = 20
    fig, ax = plt.subplots()
    ax.plot(mean_r2_test_scores[:N])
    ax2 = ax.twinx()
    _ = ax2.plot(mean_mse_test_scores[:N], c="C1")

    print(np.max(mean_r2_test_scores))

    print(mean_r2_test_scores[0])

    # XXX:
    # print("\n".join(sort_features(list(keys[0]))))

    r2_test_scores[tuple(keys[0])], np.mean(r2_test_scores[tuple(keys[0])])

    # Impact of single vegetation variable inclusion on mean scores
    for var in variable.feature_categories[variable.Category.VEGETATION]:
        vod_means = defaultdict(list)
        for i in range(6):
            for key, mean_r2 in zip(keys, mean_r2_test_scores):
                count = sum(var in feature for feature in key)
                vod_means[count].append(mean_r2)
        lengths = [len(d) for d in vod_means.values()]
        series = {
            key: pd.Series(d).reindex(range(max(lengths)))
            for key, d in vod_means.items()
        }
        var_means = pd.DataFrame(series)[list(range(6))]
        plt.figure(figsize=(15, 7))
        pd.DataFrame(var_means).boxplot()
        plt.title(var)
