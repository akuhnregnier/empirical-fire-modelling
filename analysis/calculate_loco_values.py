# -*- coding: utf-8 -*-
"""LOCO value calculation."""
import logging
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import pandas as pd
from loguru import logger as loguru_logger
from wildfires.dask_cx1 import DaskRandomForestRegressor
from wildfires.qstat import get_ncpus

from empirical_fire_modelling.analysis.loco import calculate_loco
from empirical_fire_modelling.cache import IN_STORE
from empirical_fire_modelling.configuration import (
    Experiment,
    param_dict,
    selected_features,
)
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.utils import optional_client_call

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


def loco_calc(experiment, cache_check=False, **kwargs):
    """Calculate LOCO values.

    Args:
        experiment (str): Experiment (e.g. 'ALL').
        cache_check (bool): Whether to check for cached data exclusively.

    """
    # Operate on cached data only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    loco_results = optional_client_call(
        calculate_loco,
        dict(
            rf=DaskRandomForestRegressor(**param_dict),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            leave_out=("", *selected_features[experiment]),
            local_n_jobs=(1 if (get_ncpus() < 4) else (get_ncpus() - 2)),
        ),
        cache_check=cache_check,
        add_client=True,
    )[0]

    if cache_check:
        return IN_STORE
    return loco_results


if __name__ == "__main__":
    args_loco_results = run(
        loco_calc, list(Experiment), cx1_kwargs=False, return_local_args=True
    )

    if args_loco_results is None:
        sys.exit(0)

    args, kwargs, loco_results = args_loco_results

    vis_data = {}
    for experiment, exp_results in zip(args[0], loco_results):
        for leave_out, results in exp_results.items():
            vis_data[(experiment, leave_out)] = results

    combined_df = pd.DataFrame(vis_data).T
    combined_df.index.names = ["experiment", "feature"]
    combined_df.rename(
        {
            "score": "train score",
            "mse": "train mse",
            "test_score": "test score",
            "test_mse": "test mse",
        },
        inplace=True,
        axis="columns",
    )

    loco_importances = {}
    for experiment, df in combined_df.groupby("experiment"):
        df.index = df.index.droplevel("experiment")
        reference_scores = df.loc[""]
        df = df.drop("", axis="index")
        df = reference_scores - df
        df["train mse"] = -df["train mse"]
        df["test mse"] = -df["test mse"]
        df.rename(
            {"train mse": "train -mse", "test mse": "test -mse"},
            inplace=True,
            axis="columns",
        )
        loco_importances[experiment] = df.sort_values(by="test score", ascending=False)

    pprint(loco_importances)
