# -*- coding: utf-8 -*-
"""Common variables which control the calculations and ."""
from enum import Enum
from functools import reduce
from operator import add, methodcaller
from pathlib import Path

from immutabledict import immutabledict
from wildfires.data import DATA_DIR

from . import variable

CACHE_DIR = Path(DATA_DIR) / "cache_data" / "empirical_fire_modelling"

Experiment = Enum(
    "Experiment",
    [
        "ALL",
        "TOP15",
        "CURR",
        "15VEG_FAPAR",
        "15VEG_LAI",
        "15VEG_SIF",
        "15VEG_VOD",
        "CURRDD_FAPAR",
        "CURRDD_LAI",
        "CURRDD_SIF",
        "CURRDD_VOD",
        "BEST15",
    ],
)

main_experiments = (
    Experiment.ALL,
    Experiment.TOP15,
    Experiment.CURR,
    Experiment.BEST15,
)

shared_figure_saver_kwargs = immutabledict(debug=True)
figure_saver_kwargs = immutabledict({**shared_figure_saver_kwargs, **dict(dpi=300)})
map_figure_saver_kwargs = immutabledict(
    {**shared_figure_saver_kwargs, **dict(dpi=1200)}
)
figure_save_dir = Path("~") / "tmp" / "empirical_fire_modelling"

selected_features = dict(
    {
        Experiment.ALL: (
            reduce(
                add,
                (
                    list(variable.get_shifted_variables(var_factory))
                    for var_factory in (
                        variable.DRY_DAY_PERIOD,
                        variable.SWI,
                        variable.MAX_TEMP,
                        variable.DIURNAL_TEMP_RANGE,
                        variable.LIGHTNING,
                        variable.PFT_CROP,
                        variable.POPD,
                        variable.PFT_HERB,
                        variable.SHRUB_ALL,
                        variable.TREE_ALL,
                        variable.AGB_TREE,
                        variable.VOD,
                        variable.FAPAR,
                        variable.LAI,
                        variable.SIF,
                    )
                ),
            )
        ),
        Experiment.CURR: (
            variable.DRY_DAY_PERIOD[0],
            variable.SWI[0],
            variable.MAX_TEMP[0],
            variable.DIURNAL_TEMP_RANGE[0],
            variable.LIGHTNING[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.SHRUB_ALL[0],
            variable.TREE_ALL[0],
            variable.AGB_TREE[0],
            variable.VOD[0],
            variable.FAPAR[0],
            variable.LAI[0],
            variable.SIF[0],
        ),
        Experiment.BEST15: (
            variable.DRY_DAY_PERIOD[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[6],
            variable.DRY_DAY_PERIOD[9],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.AGB_TREE[0],
            variable.VOD[9],
            variable.FAPAR[0],
            variable.FAPAR[1],
            variable.LAI[3],
            variable.SIF[6],
        ),
        Experiment.TOP15: (
            variable.DRY_DAY_PERIOD[0],
            variable.FAPAR[0],
            variable.MAX_TEMP[0],
            variable.VOD[3],
            variable.LAI[1],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.SIF[0],
            variable.LAI[3],
            variable.VOD[0],
            variable.VOD[1],
            variable.FAPAR[1],
            variable.PFT_CROP[0],
            variable.SIF[9],
            variable.POPD[0],
        ),
        Experiment["15VEG_FAPAR"]: (
            variable.DRY_DAY_PERIOD[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[6],
            variable.DRY_DAY_PERIOD[9],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.AGB_TREE[0],
            variable.FAPAR[0],
            variable.FAPAR[1],
            variable.FAPAR[3],
            variable.FAPAR[6],
            variable.FAPAR[9],
        ),
        Experiment["15VEG_LAI"]: (
            variable.DRY_DAY_PERIOD[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[6],
            variable.DRY_DAY_PERIOD[9],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.AGB_TREE[0],
            variable.LAI[0],
            variable.LAI[1],
            variable.LAI[3],
            variable.LAI[6],
            variable.LAI[9],
        ),
        Experiment["15VEG_SIF"]: (
            variable.DRY_DAY_PERIOD[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[6],
            variable.DRY_DAY_PERIOD[9],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.AGB_TREE[0],
            variable.SIF[0],
            variable.SIF[1],
            variable.SIF[3],
            variable.SIF[6],
            variable.SIF[9],
        ),
        Experiment["15VEG_VOD"]: (
            variable.DRY_DAY_PERIOD[0],
            variable.DRY_DAY_PERIOD[1],
            variable.DRY_DAY_PERIOD[3],
            variable.DRY_DAY_PERIOD[6],
            variable.DRY_DAY_PERIOD[9],
            variable.MAX_TEMP[0],
            variable.PFT_CROP[0],
            variable.POPD[0],
            variable.PFT_HERB[0],
            variable.AGB_TREE[0],
            variable.VOD[0],
            variable.VOD[1],
            variable.VOD[3],
            variable.VOD[6],
            variable.VOD[9],
        ),
        Experiment.CURRDD_FAPAR: (
            variable.DRY_DAY_PERIOD[0],
            variable.MAX_TEMP[0],
            variable.TREE_ALL[0],
            variable.SWI[0],
            variable.PFT_HERB[0],
            variable.DIURNAL_TEMP_RANGE[0],
            variable.SHRUB_ALL[0],
            variable.AGB_TREE[0],
            variable.PFT_CROP[0],
            variable.LIGHTNING[0],
            variable.FAPAR[0],
            variable.FAPAR[1],
            variable.FAPAR[3],
            variable.FAPAR[6],
            variable.FAPAR[9],
        ),
        Experiment.CURRDD_LAI: (
            variable.DRY_DAY_PERIOD[0],
            variable.MAX_TEMP[0],
            variable.TREE_ALL[0],
            variable.SWI[0],
            variable.PFT_HERB[0],
            variable.DIURNAL_TEMP_RANGE[0],
            variable.SHRUB_ALL[0],
            variable.AGB_TREE[0],
            variable.PFT_CROP[0],
            variable.LIGHTNING[0],
            variable.LAI[0],
            variable.LAI[1],
            variable.LAI[3],
            variable.LAI[6],
            variable.LAI[9],
        ),
        Experiment.CURRDD_SIF: (
            variable.DRY_DAY_PERIOD[0],
            variable.MAX_TEMP[0],
            variable.TREE_ALL[0],
            variable.SWI[0],
            variable.PFT_HERB[0],
            variable.DIURNAL_TEMP_RANGE[0],
            variable.SHRUB_ALL[0],
            variable.AGB_TREE[0],
            variable.PFT_CROP[0],
            variable.LIGHTNING[0],
            variable.SIF[0],
            variable.SIF[1],
            variable.SIF[3],
            variable.SIF[6],
            variable.SIF[9],
        ),
        Experiment.CURRDD_VOD: (
            variable.DRY_DAY_PERIOD[0],
            variable.MAX_TEMP[0],
            variable.TREE_ALL[0],
            variable.SWI[0],
            variable.PFT_HERB[0],
            variable.DIURNAL_TEMP_RANGE[0],
            variable.SHRUB_ALL[0],
            variable.AGB_TREE[0],
            variable.PFT_CROP[0],
            variable.LIGHTNING[0],
            variable.VOD[0],
            variable.VOD[1],
            variable.VOD[3],
            variable.VOD[6],
            variable.VOD[9],
        ),
    }
)
# Get the required offset features and transform the dict into and immutabledict.
selected_features = immutabledict(
    {
        exp: tuple(map(methodcaller("get_offset"), exp_vars))
        for exp, exp_vars in selected_features.items()
    }
)

assert len(selected_features) == len(
    Experiment
), "There should be as many experiment feature specs as experiments."
assert all(
    isinstance(exp, Experiment) for exp in selected_features
), "All keys should be Experiment instances."
assert all(
    all(isinstance(var, variable.Variable) for var in selected)
    for selected in selected_features.values()
), "All variables should be variable.Variable instances."

# SHAP parameters.
# NOTE: original value 2000 (~6 hrs per job?)
shap_job_samples = 2000  # Samples per job.

shap_interact_params = immutabledict(
    job_samples=50,  # Samples per job.
    max_index=5999,  # Maximum job array index (inclusive).
)

# Specify common RF (training) params.
n_splits = 5

default_param_dict = immutabledict(random_state=1, bootstrap=True, oob_score=True)

# XXX: Debug parameters!
param_dict = immutabledict(
    {**dict(max_depth=15, n_estimators=50), **default_param_dict}
)

# Training and validation test splitting.
train_test_split_kwargs = immutabledict(random_state=1, shuffle=True, test_size=0.3)
