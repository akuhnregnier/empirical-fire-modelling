# -*- coding: utf-8 -*-
"""Mean GFED4 variogram plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger
from wildfires.qstat import get_ncpus
from wildfires.variogram import plot_variogram

from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data.gfed4_variogram_data import get_gfed4_variogram_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.plotting import (
    figure_saver,
    format_label_string_with_exponent,
)

mpl.rc_file(Path(__file__).resolve().parent.parent / "matplotlibrc")

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


def gfed4_variogram(i):
    chosen_coords, chosen_ba_data, title = get_gfed4_variogram_data(i)

    fig, ax1, ax2 = plot_variogram(
        chosen_coords,
        chosen_ba_data,
        bins=50,
        max_lag=2000,
        n_jobs=get_ncpus(),
        n_per_job=6000,
        verbose=True,
    )
    # fig.suptitle(f"{title}, {inds.shape[0]} samples (out of {valid_indices.shape[0]})")
    ax1.set_ylabel("Semivariance")
    ax2.set_ylabel("N")
    ax2.set_yscale("log")
    ax1.set_xlabel("Lag (km)")

    for ax in (ax1, ax2):
        ax.grid()

    format_label_string_with_exponent(ax1, axis="y")

    fig.align_labels()

    figure_saver.save_figure(fig, "mean_gfed4_variogram")


def plot_mean_gfed4_variogram(*args, **kwargs):
    gfed4_variogram(-1)


if __name__ == "__main__":
    cx1_kwargs = dict(ncpus=1, walltime="24:00:00", memory="10GB")
    run(plot_mean_gfed4_variogram, [None], cx1_kwargs=cx1_kwargs)
