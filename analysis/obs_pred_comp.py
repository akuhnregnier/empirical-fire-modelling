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
from wildfires.utils import simple_sci_format

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import get_data, get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, threading_get_model_predict
from empirical_fire_modelling.plotting import cube_plotting, figure_saver
from empirical_fire_modelling.utils import get_mm_data

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

    get_data(experiment, cache_check=True)
    master_mask = get_data(experiment)[2]

    u_pre = threading_get_model_predict(
        X_train=X_train,
        y_train=y_train,
        predict_X=X_test,
    )

    masked_val_data = get_mm_data(y_val.values, master_mask, "val")
    predicted_ba = get_mm_data(u_pre, master_mask, "val")

    with figure_saver(sub_directory=experiment.name)(
        f"{experiment.name}_obs_pred_comp", sub_directory="predictions"
    ):
        cube_plotting(
            np.mean(masked_val_data - predicted_ba, axis=0),
            fig=plt.figure(figsize=(5.1, 2.3)),
            cmap="BrBG",
            cmap_midpoint=0,
            cmap_symmetric=False,
            boundaries=[-0.01, -0.001, -1e-4, 0, 0.001, 0.01, 0.02],
            colorbar_kwargs=dict(
                format=ticker.FuncFormatter(lambda x, pos: simple_sci_format(x)),
                pad=0.02,
                label="Ob. - Pr.",
            ),
            title="",
            coastline_kwargs={"linewidth": 0.3},
        )


if __name__ == "__main__":
    run(plot_obs_pred_comp, list(Experiment), cx1_kwargs=False)
