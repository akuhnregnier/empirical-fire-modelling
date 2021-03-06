# -*- coding: utf-8 -*-
"""SHAP value calculation."""
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger

from empirical_fire_modelling.analysis.shap import get_shap_params, get_shap_values
from empirical_fire_modelling.cache import check_in_store
from empirical_fire_modelling.configuration import all_experiments, param_dict
from empirical_fire_modelling.cx1 import parse_args, run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import call_get_model_check_cache

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

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


def shap_values(experiment, index, cache_check=False, **kwargs):
    if cache_check:
        check_in_store(get_experiment_split_data, experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    shap_params = get_shap_params(X_train)

    rf, client = call_get_model_check_cache(
        X_train, y_train, param_dict, cache_check=cache_check
    )

    calc_shap_args = (
        rf,
        X_train[
            index
            * shap_params["job_samples"] : (index + 1)
            * shap_params["job_samples"]
        ],
    )

    if cache_check:
        return check_in_store(get_shap_values, *calc_shap_args)

    return get_shap_values(*calc_shap_args)


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="06:00:00", ncpus=1, mem="7GB")

    args = [[], []]

    for experiment in tqdm(
        all_experiments[: 1 if parse_args().single else None],
        desc="Preparing SHAP arguments",
        disable=not parse_args().verbose,
    ):
        N = get_shap_params(get_experiment_split_data(experiment)[0])["max_index"] + 1
        indices = np.arange(N)
        args[0].extend([experiment] * N)
        args[1].extend(indices)

    run(shap_values, *args, cx1_kwargs=cx1_kwargs)
