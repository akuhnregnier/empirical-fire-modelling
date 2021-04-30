# -*- coding: utf-8 -*-
"""Binned BA plotting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_endog_exog_mask, get_experiment_split_data
from empirical_fire_modelling.data.cached_processing import get_obs_pred_diff_cube
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import (
    disc_cube_plot,
    get_aux0_aux1_kwargs,
    get_sci_format,
    map_figure_saver,
)
from empirical_fire_modelling.utils import check_master_masks

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


def plot_obs_pred_comp(experiment, **kwargs):
    # Operate on cached data/models only.
    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_val = get_experiment_split_data(experiment)
    get_model(X_train, y_train, cache_check=True)

    get_endog_exog_mask.check_in_store(experiment)
    master_mask = get_endog_exog_mask(experiment)[2]

    check_master_masks(master_mask)

    u_pre = threading_get_model_predict(
        X_train=X_train,
        y_train=y_train,
        predict_X=X_test,
    )

    obs_pred_diff_cube = get_obs_pred_diff_cube(y_val, u_pre, master_mask)

    with map_figure_saver(sub_directory=experiment.name)(
        f"{experiment.name}_obs_pred_comp", sub_directory="predictions"
    ):
        disc_cube_plot(
            obs_pred_diff_cube,
            fig=plt.figure(figsize=(5.1, 2.3)),
            cmap="BrBG",
            cmap_midpoint=0,
            cmap_symmetric=False,
            bin_edges=[-0.01, -0.001, -1e-4, 0, 0.001, 0.01, 0.02],
            extend="both",
            cbar_format=get_sci_format(ndigits=0),
            cbar_pad=0.025,
            cbar_label="Ob. - Pr.",
            **get_aux0_aux1_kwargs(y_val, master_mask),
            loc=(0.83, 0.14),
        )


if __name__ == "__main__":
    run(plot_obs_pred_comp, list(Experiment), cx1_kwargs=False)
