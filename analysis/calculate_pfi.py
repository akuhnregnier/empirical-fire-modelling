# -*- coding: utf-8 -*-
"""SHAP value calculation."""
import logging
import os
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
from loguru import logger as loguru_logger
from wildfires.dask_cx1 import get_client

from empirical_fire_modelling.analysis.pfi import calculate_pfi
from empirical_fire_modelling.cache import check_in_store
from empirical_fire_modelling.configuration import all_experiments, param_dict
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model

if "TQDMAUTO" in os.environ:
    pass
else:
    pass

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


def pfi_calc(experiment, cache_check=False, **kwargs):
    """Calculate PFIs for both training and test data.

    Args:
        experiment (str): Experiment (e.g. 'ALL').
        data ({'test', 'train'}): Which data to use.
        cache_check (bool): Whether to check for cached data exclusively.

    """
    if cache_check:
        check_in_store(get_experiment_split_data, experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    if cache_check:
        check_in_store(get_model, X_train, y_train, param_dict)

    # Get Dask client that is needed for model fitting.
    client = get_client(fallback=True, fallback_threaded=True)

    rf = get_model(X_train, y_train, param_dict)

    # Test data.
    pfi_test_args = (rf, X_test, y_test)
    if cache_check:
        check_in_store(calculate_pfi, *pfi_test_args)

    # Train data.
    pfi_train_args = (rf, X_train, y_train)
    if cache_check:
        return check_in_store(calculate_pfi, *pfi_train_args)

    return {
        "train": calculate_pfi(*pfi_train_args),
        "test": calculate_pfi(*pfi_test_args),
    }


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="06:00:00", ncpus=32, mem="60GB")

    pprint(run(pfi_calc, all_experiments, cx1_kwargs=cx1_kwargs))
