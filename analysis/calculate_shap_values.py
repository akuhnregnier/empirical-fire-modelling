# -*- coding: utf-8 -*-
"""SHAP value calculation."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger

from empirical_fire_modelling.analysis.shap import get_shap_params, get_shap_values
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import parse_args, run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.utils import tqdm

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
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    shap_params = get_shap_params(X_train)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    rf = get_model(X_train, y_train)

    calc_shap_args = (
        rf,
        X_train[
            index
            * shap_params["job_samples"] : (index + 1)
            * shap_params["job_samples"]
        ],
    )

    if cache_check:
        return get_shap_values.check_in_store(*calc_shap_args)

    return get_shap_values(*calc_shap_args)


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="06:00:00", ncpus=1, mem="7GB")

    args = [[], []]
    experiments = list(Experiment)
    chosen_experiments = experiments[: 1 if parse_args().single else None]

    for experiment in tqdm(
        chosen_experiments,
        desc="Preparing SHAP arguments",
        disable=not parse_args().verbose,
    ):
        N = get_shap_params(get_experiment_split_data(experiment)[0])["max_index"] + 1
        indices = np.arange(N)
        args[0].extend([experiment] * N)
        args[1].extend(indices)

    raw_shap_data = run(shap_values, *args, cx1_kwargs=cx1_kwargs)

    shap_data = {}
    for ((experiment, index), data) in zip(zip(*args), raw_shap_data):
        shap_data[(experiment, index)] = data

    # Join data for the different experiments.
    joined_data = {}
    for experiment in chosen_experiments:
        selected_shap_data = [
            data for ((exp, index), data) in shap_data.items() if exp == experiment
        ]
        joined_data[experiment] = np.vstack(selected_shap_data)
