# -*- coding: utf-8 -*-
"""Common variables which control the calculations and ."""
import re
from pathlib import Path

from wildfires.utils import shorten_features


def get_offset_features(features):
    """Replace large offsets with their transformed representation."""
    if features is None:
        return None

    offset_features = []
    for column in features:
        match = re.search(r"-\d{1,2}", column)
        if match:
            span = match.span()
            # Change the string to reflect the shift.
            original_offset = int(column[slice(*span)])
            if original_offset > -12:
                # Only shift months that are 12 or more months before the current month.
                offset_features.append(column)
                continue
            comp = -(-original_offset % 12)
            new_column = " ".join(
                (
                    column[: span[0] - 1],
                    f"{original_offset} - {comp}",
                    column[span[1] + 1 :],
                )
            )
            offset_features.append(new_column)
        else:
            offset_features.append(column)

    return offset_features


shared_figure_saver_kwargs = dict(debug=True)
figure_saver_kwargs = {**shared_figure_saver_kwargs, **dict(dpi=300)}
map_figure_saver_kwargs = {**shared_figure_saver_kwargs, **dict(dpi=1200)}
figure_save_dir = Path("~") / "tmp" / "empirical_fire_modelling"

# Investigated lags.
lags = (0, 1, 3, 6, 9, 12, 18, 24)

# Data filling params.
st_persistent_perc = 50
st_k = 4

filled_variables = {"SWI(1)", "FAPAR", "LAI", "VOD Ku-band", "SIF"}
filled_variables.update(shorten_features(filled_variables))


def fill_name(name):
    return f"{name} {st_persistent_perc}P {st_k}k"


def get_filled_names(names):
    if isinstance(names, str):
        return get_filled_names((names,))[0]
    filled = []
    for name in names:
        if any(var in name for var in filled_variables):
            filled.append(fill_name(name))
        else:
            filled.append(name)
    return filled


main_experiments = ["all", "15_most_important", "no_temporal_shifts", "best_top_15"]

experiment_name_dict = {
    "all": "ALL",
    "15_most_important": "TOP15",
    "no_temporal_shifts": "CURR",
    "fapar_only": "15VEG_FAPAR",
    "lai_only": "15VEG_LAI",
    "sif_only": "15VEG_SIF",
    "vod_only": "15VEG_VOD",
    "lagged_fapar_only": "CURRDD_FAPAR",
    "lagged_lai_only": "CURRDD_LAI",
    "lagged_sif_only": "CURRDD_SIF",
    "lagged_vod_only": "CURRDD_VOD",
    "best_top_15": "BEST15",
}

all_experiments = list(experiment_name_dict)

selected_features = {
    "ALL": None,  # Sentinel value indicating all values are selected.
    "CURR": tuple(
        get_filled_names(
            (
                "Dry Day Period",
                "SWI(1)",
                "Max Temp",
                "Diurnal Temp Range",
                "lightning",
                "pftCrop",
                "popd",
                "pftHerb",
                "ShrubAll",
                "TreeAll",
                "AGB Tree",
                "VOD Ku-band",
                "FAPAR",
                "LAI",
                "SIF",
            )
        )
    ),
    "BEST15": (
        "Dry Day Period",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "Dry Day Period -6 Month",
        "Dry Day Period -9 Month",
        "Max Temp",
        "pftCrop",
        "popd",
        "pftHerb",
        "AGB Tree",
        "VOD Ku-band 50P 4k -9 Month",
        "FAPAR 50P 4k",
        "FAPAR 50P 4k -1 Month",
        "LAI 50P 4k -3 Month",
        "SIF 50P 4k -6 Month",
    ),
    "TOP15": (
        "Dry Day Period",
        "FAPAR 50P 4k",
        "Max Temp",
        "VOD Ku-band 50P 4k -3 Month",
        "LAI 50P 4k -1 Month",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "SIF 50P 4k",
        "LAI 50P 4k -3 Month",
        "VOD Ku-band 50P 4k -1 Month",
        "VOD Ku-band 50P 4k",
        "FAPAR 50P 4k -1 Month",
        "pftCrop",
        "SIF 50P 4k -9 Month",
        "popd",
    ),
    "15VEG_FAPAR": (
        "Dry Day Period",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "Dry Day Period -6 Month",
        "Dry Day Period -9 Month",
        "Max Temp",
        "pftCrop",
        "popd",
        "pftHerb",
        "AGB Tree",
        "FAPAR 50P 4k",
        "FAPAR 50P 4k -1 Month",
        "FAPAR 50P 4k -3 Month",
        "FAPAR 50P 4k -6 Month",
        "FAPAR 50P 4k -9 Month",
    ),
    "15VEG_LAI": (
        "Dry Day Period",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "Dry Day Period -6 Month",
        "Dry Day Period -9 Month",
        "Max Temp",
        "pftCrop",
        "popd",
        "pftHerb",
        "AGB Tree",
        "LAI 50P 4k",
        "LAI 50P 4k -1 Month",
        "LAI 50P 4k -3 Month",
        "LAI 50P 4k -6 Month",
        "LAI 50P 4k -9 Month",
    ),
    "15VEG_SIF": (
        "Dry Day Period",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "Dry Day Period -6 Month",
        "Dry Day Period -9 Month",
        "Max Temp",
        "pftCrop",
        "popd",
        "pftHerb",
        "AGB Tree",
        "SIF 50P 4k",
        "SIF 50P 4k -1 Month",
        "SIF 50P 4k -3 Month",
        "SIF 50P 4k -6 Month",
        "SIF 50P 4k -9 Month",
    ),
    "15VEG_VOD": (
        "Dry Day Period",
        "Dry Day Period -1 Month",
        "Dry Day Period -3 Month",
        "Dry Day Period -6 Month",
        "Dry Day Period -9 Month",
        "Max Temp",
        "pftCrop",
        "popd",
        "pftHerb",
        "AGB Tree",
        "VOD Ku-band 50P 4k",
        "VOD Ku-band 50P 4k -1 Month",
        "VOD Ku-band 50P 4k -3 Month",
        "VOD Ku-band 50P 4k -6 Month",
        "VOD Ku-band 50P 4k -9 Month",
    ),
    "CURRDD_FAPAR": (
        "Dry Day Period",
        "Max Temp",
        "TreeAll",
        "SWI(1) 50P 4k",
        "pftHerb",
        "Diurnal Temp Range",
        "ShrubAll",
        "AGB Tree",
        "pftCrop",
        "lightning",
        "FAPAR 50P 4k",
        "FAPAR 50P 4k -1 Month",
        "FAPAR 50P 4k -3 Month",
        "FAPAR 50P 4k -6 Month",
        "FAPAR 50P 4k -9 Month",
    ),
    "CURRDD_LAI": (
        "Dry Day Period",
        "Max Temp",
        "TreeAll",
        "SWI(1) 50P 4k",
        "pftHerb",
        "Diurnal Temp Range",
        "ShrubAll",
        "AGB Tree",
        "pftCrop",
        "lightning",
        "LAI 50P 4k",
        "LAI 50P 4k -1 Month",
        "LAI 50P 4k -3 Month",
        "LAI 50P 4k -6 Month",
        "LAI 50P 4k -9 Month",
    ),
    "CURRDD_SIF": (
        "Dry Day Period",
        "Max Temp",
        "TreeAll",
        "SWI(1) 50P 4k",
        "pftHerb",
        "Diurnal Temp Range",
        "ShrubAll",
        "AGB Tree",
        "pftCrop",
        "lightning",
        "SIF 50P 4k",
        "SIF 50P 4k -1 Month",
        "SIF 50P 4k -3 Month",
        "SIF 50P 4k -6 Month",
        "SIF 50P 4k -9 Month",
    ),
    "CURRDD_VOD": (
        "Dry Day Period",
        "Max Temp",
        "TreeAll",
        "SWI(1) 50P 4k",
        "pftHerb",
        "Diurnal Temp Range",
        "ShrubAll",
        "AGB Tree",
        "pftCrop",
        "lightning",
        "VOD Ku-band 50P 4k",
        "VOD Ku-band 50P 4k -1 Month",
        "VOD Ku-band 50P 4k -3 Month",
        "VOD Ku-band 50P 4k -6 Month",
        "VOD Ku-band 50P 4k -9 Month",
    ),
}

offset_selected_features = {
    exp: get_offset_features(features) for exp, features in selected_features.items()
}

assert set(offset_selected_features) == set(
    experiment_name_dict.values()
), "All experiments should define their selected features."


units = {
    "DD": "days",
    "SWI": r"$\mathrm{m}^3 \mathrm{m}^{-3}$",
    "MaxT": "K",
    "DTR": "K",
    "Lightning": r"$\mathrm{strokes}\ \mathrm{km}^{-2}$",
    "CROP": "1",
    "POPD": r"$\mathrm{inh}\ \mathrm{km}^{-2}$",
    "HERB": "1",
    "SHRUB": "1",
    "TREE": "1",
    "AGB": "r$\mathrm{kg}\ \mathrm{m}^{-2}$",
    "VOD": "1",
    "FAPAR": "1",
    "LAI": r"$\mathrm{m}^2\ \mathrm{m}^{-2}$",
    "SIF": "r$\mathrm{mW}\ \mathrm{m}^{-2}\ \mathrm{sr}^{-1}\ \mathrm{nm}^{-1}$",
}

# SHAP parameters.
# XXX: original value 2000 (~6 hrs per job?)
shap_job_samples = 20  # Samples per job.

shap_interact_params = {
    "job_samples": 50,  # Samples per job.
    "max_index": 5999,  # Maximum job array index (inclusive).
}

# Specify common RF (training) params.
n_splits = 5

default_param_dict = {"random_state": 1, "bootstrap": True, "oob_score": True}

param_dict = {**default_param_dict}

# Training and validation test splitting.
train_test_split_kwargs = dict(random_state=1, shuffle=True, test_size=0.3)

feature_categories = {
    # `get_filled_names` may need to be called here if needed.
    "meteorology": [
        "Dry Day Period",
        "SWI(1)",
        "Max Temp",
        "Diurnal Temp Range",
        "lightning",
    ],
    "human": ["pftCrop", "popd"],
    "landcover": ["pftHerb", "ShrubAll", "TreeAll", "AGB Tree"],
    "vegetation": ["VOD Ku-band", "FAPAR", "LAI", "SIF"],
}

feature_order = {}
no_fill_feature_order = {}
counter = 0
for category, entries in feature_categories.items():
    for entry in entries:
        feature_order[entry] = counter
        no_fill_feature_order[entry.strip(fill_name(""))] = counter
        counter += 1
        no_fill_feature_order[shorten_features(entry.strip(fill_name("")))] = counter
        counter += 1

# If BA is included, position it first.
no_fill_feature_order["GFED4 BA"] = -1
no_fill_feature_order["BA"] = -2
