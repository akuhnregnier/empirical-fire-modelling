# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns

from ..configuration import Experiment, main_experiments
from ..variable import lags

# Colors.
experiment_colors = sns.color_palette("Set2")
experiment_color_dict = {
    **{
        experiment: color
        for experiment, color in zip(main_experiments, experiment_colors)
    },
    Experiment["15VEG_FAPAR"]: experiment_colors[4],
    Experiment["15VEG_LAI"]: experiment_colors[4],
    Experiment["15VEG_SIF"]: experiment_colors[4],
    Experiment["15VEG_VOD"]: experiment_colors[4],
    Experiment["CURRDD_FAPAR"]: experiment_colors[5],
    Experiment["CURRDD_LAI"]: experiment_colors[5],
    Experiment["CURRDD_SIF"]: experiment_colors[5],
    Experiment["CURRDD_VOD"]: experiment_colors[5],
}

# 9 colors used to differentiate varying the lags throughout.
lag_colors = sns.color_palette("Set1", desat=0.85)
lag_color_dict = {lag: color for lag, color in zip(lags, lag_colors)}

# Markers.
experiment_markers = ["<", "o", ">", "x"]
experiment_marker_dict = {
    **{
        experiment: marker
        for experiment, marker in zip(main_experiments, experiment_markers)
    },
    Experiment["15VEG_FAPAR"]: "|",
    Experiment["15VEG_LAI"]: "|",
    Experiment["15VEG_SIF"]: "|",
    Experiment["15VEG_VOD"]: "|",
    Experiment["CURRDD_FAPAR"]: "^",
    Experiment["CURRDD_LAI"]: "^",
    Experiment["CURRDD_SIF"]: "^",
    Experiment["CURRDD_VOD"]: "^",
}

# Zorders.
experiment_zorder_dict = {
    Experiment["ALL"]: 7,
    Experiment["TOP15"]: 6,
    Experiment["CURR"]: 5,
    Experiment["BEST15"]: 4,
    Experiment["15VEG_FAPAR"]: 3,
    Experiment["15VEG_LAI"]: 3,
    Experiment["15VEG_SIF"]: 3,
    Experiment["15VEG_VOD"]: 3,
    Experiment["CURRDD_FAPAR"]: 2,
    Experiment["CURRDD_LAI"]: 2,
    Experiment["CURRDD_SIF"]: 2,
    Experiment["CURRDD_VOD"]: 2,
}

plotting_experiments = set(experiment_color_dict)
assert (
    plotting_experiments == set(experiment_marker_dict) == set(experiment_zorder_dict)
)

# Combined plotting kwargs.
experiment_plot_kwargs = {
    experiment: {
        "label": experiment.name,
        "c": experiment_color_dict[experiment],
        "marker": experiment_marker_dict[experiment],
        "zorder": experiment_zorder_dict[experiment],
    }
    for experiment in plotting_experiments
}


aux0_c = np.array([150, 150, 150, 200], dtype=np.float64) / 255
aux1_c = np.array([64, 64, 64, 200], dtype=np.float64) / 255
