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

from empirical_fire_modelling.analysis.loco import calculate_loco
from empirical_fire_modelling.cache import IN_STORE
from empirical_fire_modelling.configuration import Experiment, selected_features
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
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

    # Operate on cached fitted models only.
    get_model(X_train, y_train, cache_check=True)
    rf = get_model(X_train, y_train)

    loco_results = optional_client_call(
        calculate_loco,
        dict(
            rf=rf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            leave_out=("", *selected_features[experiment]),
            local_n_jobs=1,
        ),
        cache_check=cache_check,
        add_client=True,
    )[0]

    if cache_check:
        return IN_STORE
    return loco_results


if __name__ == "__main__":
    experiments = list(Experiment)
    loco_results = run(loco_calc, experiments, cx1_kwargs=False)

    vis_data = {}
    for experiment, exp_results in zip(experiments, loco_results):
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
        loco_importances[experiment] = df.sort_values(by="test score", ascending=False)

    pprint(loco_importances)
