# -*- coding: utf-8 -*-
"""SHAP value calculation."""
import logging
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.analysis.pfi import calculate_pfi
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model

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


def pfi_calc(experiment, cache_check=False, **kwargs):
    """Calculate PFIs for both training and test data.

    Args:
        experiment (str): Experiment (e.g. 'ALL').
        data ({'test', 'train'}): Which data to use.
        cache_check (bool): Whether to check for cached data exclusively.

    """
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    rf = get_model(X_train, y_train)

    # Test data.
    pfi_test_args = (rf, X_test, y_test)
    if cache_check:
        calculate_pfi.check_in_store(*pfi_test_args)

    # Train data.
    pfi_train_args = (rf, X_train, y_train)
    if cache_check:
        return calculate_pfi.check_in_store(*pfi_train_args)

    return {
        "train": calculate_pfi(*pfi_train_args),
        "test": calculate_pfi(*pfi_test_args),
    }


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="24:00:00", ncpus=32, mem="60GB")

    experiments = list(Experiment)
    args_pfi_results = run(
        pfi_calc, experiments, cx1_kwargs=cx1_kwargs, return_local_args=True
    )
    if args_pfi_results is None:
        sys.exit(0)

    args, kwargs, pfi_results = args_pfi_results

    pfi_importances = {}
    for exp, pfi_result in zip(args[0], pfi_results):
        # Join the train and test data.
        pfi_importances[exp] = (
            pfi_result["train"]
            .set_index("feature", drop=True)
            .rename({"weight": "train weight", "std": "train std"}, axis="columns")
            .join(
                pfi_result["test"]
                .set_index("feature", drop=True)
                .rename({"weight": "test weight", "std": "test std"}, axis="columns")
            )
        ).sort_values("test weight", ascending=False)

    pprint(pfi_importances)
