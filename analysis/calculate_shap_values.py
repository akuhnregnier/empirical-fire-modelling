# -*- coding: utf-8 -*-
"""SHAP value calculation."""
import logging
import sys
import warnings
from operator import attrgetter
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import numpy as np
import pandas as pd
from loguru import logger as loguru_logger

from empirical_fire_modelling.analysis.shap import get_shap_params, get_shap_values
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import get_parsers, run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.exceptions import NotCachedError
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.utils import tqdm

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
        X_train.iloc[
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

    cmd_args = get_parsers()["parser"].parse_args()

    if cmd_args.experiment is not None:
        chosen_experiments = [
            exp for exp in experiments if exp == Experiment[cmd_args.experiment]
        ]
    else:
        chosen_experiments = experiments.copy()

    chosen_experiments = chosen_experiments[: 1 if cmd_args.single else None]

    run_experiments = []
    for experiment in chosen_experiments:
        try:
            # Check if a full cache is already present.
            # Operate on cached data only.
            get_experiment_split_data.check_in_store(experiment)
            X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

            # Operate on cached fitted models only.
            get_model(X_train, y_train, cache_check=True)
            rf = get_model(X_train, y_train)

            get_shap_values.check_in_store(rf, X_train)
        except NotCachedError:
            run_experiments.append(experiment)

    for experiment in tqdm(
        run_experiments,
        desc="Preparing SHAP arguments",
        disable=not cmd_args.verbose,
    ):
        N = get_shap_params(get_experiment_split_data(experiment)[0])["max_index"] + 1
        indices = np.arange(N)
        args[0].extend([experiment] * N)
        args[1].extend(indices)

    raw_shap_data = run(shap_values, *args, cx1_kwargs=cx1_kwargs)

    if raw_shap_data is None:
        if run_experiments:
            # Experiments were submitted as CX1 jobs.
            sys.exit(0)
        # Otherwise, experiments were already present as a fully cached value.

    if isinstance(raw_shap_data, dict) and set(raw_shap_data) == {
        "present",
        "uncached",
    }:
        # Checking was performed.
        print("Full cache present for:", end="")
        pprint(
            set(map(attrgetter("name"), set(chosen_experiments) - set(run_experiments)))
        )
        sys.exit(0)

    # Load all data, which is faster using `get_shap_values()` along with all
    # data to cache the loading and concatenation of the individual entries.
    experiment_shap_data = {}
    for experiment in tqdm(
        chosen_experiments,
        desc="Loading joined SHAP data",
        disable=not cmd_args.verbose,
    ):
        X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)
        rf = get_model(X_train, y_train)
        experiment_shap_data[experiment] = get_shap_values(rf, X_train)

    shap_importances = {}
    for exp, shap_arr in tqdm(experiment_shap_data.items()):
        X_train, X_test, y_train, y_test = get_experiment_split_data(exp)
        abs_shap_values = np.abs(shap_arr)
        agg_df = pd.DataFrame(
            {
                "mean SHAP": np.mean(abs_shap_values, axis=0),
                "std SHAP": np.std(abs_shap_values, axis=0),
            },
            index=map(str, X_train.columns),
        )
        shap_importances[exp] = agg_df.sort_values("mean SHAP", ascending=False)
