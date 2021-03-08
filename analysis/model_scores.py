# -*- coding: utf-8 -*-
"""Retrieving model scores. Expected to be run locally."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment, param_dict
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, get_model_scores

mpl.rc_file(Path(__file__).resolve().parent / "matplotlibrc")

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging(level="DEBUG")

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)


def get_experiment_model_scores(experiment, cache_check=False, **kwargs):
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    # Operate on cached fitted models only.
    get_model(X_train, y_train, param_dict, cache_check=True)
    model = get_model(X_train, y_train, param_dict)

    if cache_check:
        return get_model_scores.check_in_store(model, X_test, X_train, y_test, y_train)
    return get_model_scores(model, X_test, X_train, y_test, y_train)


if __name__ == "__main__":
    scores = run(get_experiment_model_scores, list(Experiment), cx1_kwargs=False)

    from pprint import pprint

    pprint(scores)
