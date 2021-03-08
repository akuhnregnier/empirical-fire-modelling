# -*- coding: utf-8 -*-
"""LOCO value calculation."""
import logging
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
from loguru import logger as loguru_logger
from wildfires.dask_cx1 import get_client

from empirical_fire_modelling.analysis.loco import calculate_loco
from empirical_fire_modelling.configuration import (
    Experiment,
    param_dict,
    selected_features,
)
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model

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


def loco_calc(experiment, leave_out, cache_check=False, **kwargs):
    """Calculate LOCO values.

    Args:
        experiment (str): Experiment (e.g. 'ALL').
        leave_out (iterable of column names): Column names to exclude. Empty string
            for no excluded columns (i.e. the baseline with all columns).
        cache_check (bool): Whether to check for cached data exclusively.

    """
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, param_dict, cache_check=True)
    rf = get_model(X_train, y_train, param_dict)

    client = get_client(fallback=True, fallback_threaded=True)

    loco_kwargs = dict(
        rf=rf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client=client,
        leave_out=("", *selected_features[experiment]),
        local_n_jobs=1,
    )
    if cache_check:
        return calculate_loco.check_in_store(**loco_kwargs)
    return calculate_loco(**loco_kwargs)


if __name__ == "__main__":
    pprint(run(loco_calc, list(Experiment), cx1_kwargs=False))
