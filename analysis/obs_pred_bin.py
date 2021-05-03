# -*- coding: utf-8 -*-
"""Binned BA plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as loguru_logger
from matplotlib import ticker
from matplotlib.colors import LogNorm

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import figure_saver, get_sci_format

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


def plot_obs_pred_bin(experiment, **kwargs):
    # Operate on cached data/models only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, u_val = get_experiment_split_data(experiment)
    get_model(X_train, y_train, cache_check=True)

    u_pre = threading_get_model_predict(
        X_train=X_train,
        y_train=y_train,
        predict_X=X_test,
    )

    min_non_zero_val = u_val[u_val > 0].min()

    x_edges = np.append(0, np.geomspace(min_non_zero_val, 1, 100))
    y_edges = np.geomspace(np.min(u_pre), np.max(u_pre), 100 + 1)

    h = np.histogram2d(u_val, u_pre, bins=[x_edges, y_edges])[0]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    img = ax.pcolor(
        x_edges,
        y_edges,
        h.T,
        norm=LogNorm(),
    )

    # Plot diagonal 1:1 line.
    plt.plot(
        *(
            (
                np.geomspace(
                    max(min(u_val), min(u_pre)), min(max(u_val), max(u_pre)), 200
                ),
            )
            * 2
        ),
        linestyle="--",
        c="C3",
        lw=2,
    )

    ax.set_xscale(
        "symlog", linthresh=min_non_zero_val, linscale=2e-1, subs=range(2, 10)
    )
    ax.set_yscale("log")

    def offset_sci_format(x, *args, **kwargs):
        canon = get_sci_format(ndigits=0, trim_leading_one=True)(x, None)
        if np.isclose(x, 1e-5):
            return " " * 6 + canon
        elif np.isclose(x, 0):
            return canon + " " * 3
        return canon

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: offset_sci_format(x))
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(get_sci_format(ndigits=0, trim_leading_one=True))
    )

    ax.set_xlabel("Observed (BA)")
    ax.set_ylabel("Predicted (BA)")

    ax.set_axisbelow(True)
    ax.grid(zorder=0)

    fig.colorbar(
        img,
        shrink=0.7,
        aspect=30,
        format=get_sci_format(ndigits=0, trim_leading_one=True),
        pad=0.02,
        label="samples",
    )
    figure_saver(sub_directory=experiment.name).save_figure(
        plt.gcf(), f"{experiment.name}_obs_pred_bin", sub_directory="predictions"
    )


if __name__ == "__main__":
    run(plot_obs_pred_bin, list(Experiment), cx1_kwargs=False)
