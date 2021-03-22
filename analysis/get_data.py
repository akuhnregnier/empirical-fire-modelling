# -*- coding: utf-8 -*-
"""Data retrieval."""
import logging
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment
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


def get_experiment_data(experiment, cache_check=False, **kwargs):
    if cache_check:
        get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    cx1_kwargs = dict(walltime="04:00:00", ncpus=32, mem="60GB")
    experiments = list(Experiment)
    experiment_data = dict(
        zip(
            experiments,
            run(get_experiment_data, experiments, cx1_kwargs=cx1_kwargs),
        )
    )
    for (experiment, (X_train, X_test, y_train, y_test)) in experiment_data.items():
        print(f"{experiment} → {y_train.name}")
        pprint(X_train.columns)
        print()
