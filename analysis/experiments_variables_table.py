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
from wildfires.utils import shorten_features

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
    df.rename(
        {vf: shorten_features(vf) for vf in df.columns}, axis="columns", inplace=True
    )

    template_spaces = [19, 18, 4, 5, 4, 11, 5, 5, 5, 6, 5, 4, 18, 18, 18, 22]

    formatted_rows = []
    for row in filter(None, df.to_latex().split("\n")):
        if not any(
            element in row
            for element in ("tabular", "toprule", "midrule", "bottomrule")
        ):
            split = row.split(" &")
            assert len(split) == len(template_spaces)
            # Pad each of the elements in `split` to match the desired number of
            # spaces.
            row = "& ".join(
                format(x.strip(), f"<{n}") for x, n in zip(split, template_spaces)
            )
        formatted_rows.append(row.strip())

    latex_df = "\n".join(formatted_rows) + "\n"
    assert "l" * len(template_spaces) in latex_df
    latex_df = latex_df.replace(
        "l" * len(template_spaces),
        r"L{2.2cm}L{\acols}lllL{0.6cm}llllllL{\acols}L{\acols}L{\acols}L{\acols}",
    )
    latex_df = latex_df.replace("toprule", "tophline")
    latex_df = latex_df.replace("midrule", "middlehline")
    latex_df = latex_df.replace("bottomrule", "bottomhline")
    latex_df = latex_df.replace("Lightning ", "Light-ning")
    latex_df = latex_df.replace(r"15VEG\_FAPAR\_MON ", r"15VEG\_FAPAR-\_MON")
    latex_df = latex_df.replace("all A", "all~A")

    print(latex_df)
