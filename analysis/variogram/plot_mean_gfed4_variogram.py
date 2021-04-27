# -*- coding: utf-8 -*-
"""Mean GFED4 variogram plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger
from wildfires.data import GFEDv4
from wildfires.qstat import get_ncpus
from wildfires.utils import ensure_datetime, get_land_mask
from wildfires.variogram import plot_variogram

from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.plotting import figure_saver

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
    gfed4 = GFEDv4()
    if i == -1:
        title = "Mean GFED4 BA"
        ba = gfed4.get_mean_dataset().cube
    else:
        ba = gfed4.cube[i]
        title = f"GFED4 BA {ensure_datetime(ba.coord('time').cell(0).point):%Y-%m}"

    ba.data.mask = ~get_land_mask()

    latitudes = ba.coord("latitude").points
    longitudes = ba.coord("longitude").points

    coords = []
    for lon in longitudes:
        for lat in latitudes:
            coords.append((lat, lon))
    coords = np.array(coords)
    ba_flat = ba.data.ravel()

    # Choose indices.
    valid_indices = np.where(~ba.data.mask.ravel())[0]
    # Random subset.
    # inds = np.random.default_rng(0).choice(valid_indices, size=(4000,))
    # All indices.
    inds = valid_indices

    # print(f"Max N:    {valid_indices.shape[0]:>10d}")
    # print(f"Chosen N: {inds.shape[0]:>10d}")

    fig, ax1, ax2 = plot_variogram(
        coords[inds],
        ba_flat.data[inds],
        bins=50,
        max_lag=2000,
        n_jobs=get_ncpus(),
        n_per_job=6000,
        verbose=True,
    )

    fig.suptitle(f"{title}, {inds.shape[0]} samples (out of {valid_indices.shape[0]})")
    ax1.set_ylabel("semivariance")
    ax2.set_ylabel("N")
    ax2.set_yscale("log")
    ax1.set_xlabel("Lag (km)")

    figure_saver.save_figure(fig, "mean_gfed4_variogram")


def plot_mean_gfed4_variogram(*args, **kwargs):
    gfed4_variogram(-1)


if __name__ == "__main__":
    cx1_kwargs = dict(ncpus=1, walltime="24:00:00", memory="10GB")
    run(plot_mean_gfed4_variogram, [None], cx1_kwargs=cx1_kwargs)
