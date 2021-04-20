# -*- coding: utf-8 -*-
"""Model fitting."""
import logging
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib as mpl
from joblib import parallel_backend
from loguru import logger as loguru_logger
from wildfires.qstat import get_ncpus

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


def fit_experiment_model(experiment, cache_check=False, **kwargs):
    if cache_check:
        get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    if cache_check:
        return get_model(X_train=X_train, y_train=y_train, cache_check=True)
    model = get_model(
        X_train=X_train,
        y_train=y_train,
        parallel_backend_call=(
            # Use local threading backend - avoid the Dask backend.
            partial(parallel_backend, "threading", n_jobs=get_ncpus())
        ),
    )
    return model


if __name__ == "__main__":
    cx1_kwargs = dict(walltime="24:00:00", ncpus=32, mem="60GB")
    args_models = run(
        fit_experiment_model,
        list(Experiment),
        cx1_kwargs=cx1_kwargs,
        return_local_args=True,
    )

    if args_models is None:
        sys.exit(0)

    args, kwargs, models = args_models

    models = {exp: fitted_model for exp, fitted_model in zip(args[0], models)}
