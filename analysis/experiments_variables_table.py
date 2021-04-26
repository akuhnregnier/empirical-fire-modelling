# -*- coding: utf-8 -*-
"""Retrieving model scores. Expected to be run locally."""
import logging
import sys
import warnings
from operator import attrgetter
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment, selected_features
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.variable import match_factory, sort_variables

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


if __name__ == "__main__":
    # Retrieve the variables used for each experiment.

    # Build the matrix representing which variables are present for each of the experiments
    condensed = {}

    var_factories = [var.parent for var in selected_features[Experiment.CURR]]

    for exp, exp_vars in selected_features.items():
        for var_factory in var_factories:
            # Find for which lags the current variable is present (if any).
            lags = [
                f"{v.shift}M".replace("0M", "C")
                for v in sort_variables(exp_vars)
                if match_factory(v, var_factory)
            ]
            if all(
                lag in lags
                for lag in ["C", "1M", "3M", "6M", "9M", "12M", "18M", "24M"]
            ):
                lags = "C & all A"
            else:
                lags = ", ".join(lags)
            condensed[(exp.name, str(var_factory))] = lags
    df = (
        pd.Series(condensed)
        .unstack()
        .reindex(
            index=list(map(attrgetter("name"), selected_features)),
            columns=list(map(str, var_factories)),
        )
    )

    print(df.to_latex())
