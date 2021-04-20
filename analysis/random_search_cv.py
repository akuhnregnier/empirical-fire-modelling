# -*- coding: utf-8 -*-
"""Model fitting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger
from wildfires.dask_cx1 import (
    DaskRandomForestRegressor,
    fit_dask_sub_est_random_search_cv,
)

from empirical_fire_modelling.configuration import (
    CACHE_DIR,
    Experiment,
    default_param_dict,
    n_splits,
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
    # Only carry out the analysis on the ALL model.
    X_train, X_test, y_train, y_test = get_experiment_split_data(Experiment.ALL)

    client = get_client(fallback=False)

    parameters_RF = {
        "n_estimators": [500, 1000],
        "max_depth": [16, 18],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["auto"],
        "ccp_alpha": np.linspace(0, 4e-9, 2),
    }

    results = fit_dask_sub_est_random_search_cv(
        DaskRandomForestRegressor(**default_param_dict),
        X_train.values,
        y_train.values,
        parameters_RF,
        client,
        n_splits=n_splits,
        max_time="18h",
        n_iter=None,
        verbose=True,
        return_train_score=True,
        refit=False,
        local_n_jobs=30,
        random_state=0,
        cache_dir=CACHE_DIR,
    )
