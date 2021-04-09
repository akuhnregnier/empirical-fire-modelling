# -*- coding: utf-8 -*-
"""Buffered leave-one-out cross validation."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import buffered_leave_one_out, get_data
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


def fit_buffered_loo_sample(
    experiment, radius, max_rad, seed, cache_check=False, **kwargs
):
    # Operate on cached data only.
    get_data(experiment, cache_check=True)
    endog_data, exog_data, master_mask = get_data(experiment)[:3]

    bloo_kwargs = dict(
        exog_data=exog_data,
        endog_data=endog_data,
        master_mask=master_mask,
        radius=radius,
        max_rad=max_rad,
        seed=seed,
        verbose=False,
        dpi=300,
    )
    if cache_check:
        return buffered_leave_one_out.check_in_store(**bloo_kwargs)
    (
        test_indices,
        n_ignored,
        n_train,
        n_hold_out,
        total_samples,
        hold_out_y,
        predicted_y,
    ) = buffered_leave_one_out(**bloo_kwargs)

    data_info = (
        test_indices,
        n_ignored,
        n_train,
        n_hold_out,
        total_samples,
    )

    return (data_info, hold_out_y, predicted_y)


if __name__ == "__main__":
    pass

    # For 40 estimators, ~25 minutes per fit operation.
    cx1_kwargs = dict(walltime="02:00:00", ncpus=1, mem="10GB")
    experiments = list(Experiment)

    max_rad = 50

    args = [[], [], [], []]
    for experiment in experiments:
        for radius in np.linspace(0, max_rad, 8):
            for seed in range(1000):
                args[0].append(experiment)
                args[1].append(radius)
                args[2].append(max_rad)
                args[3].append(seed)

    results = run(fit_buffered_loo_sample, *args, cx1_kwargs=cx1_kwargs)
