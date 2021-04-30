# -*- coding: utf-8 -*-
import numpy as np
from wildfires.data import dummy_lat_lon_cube

from empirical_fire_modelling.cache import cache

from ..utils import get_mm_data


@cache
def get_obs_pred_diff_cube(y_val, u_pre, master_mask):
    masked_val_data = get_mm_data(y_val.values, master_mask, "val")
    predicted_ba = get_mm_data(u_pre, master_mask, "val")
    return dummy_lat_lon_cube(np.mean(masked_val_data - predicted_ba, axis=0))


@cache
def get_ba_plotting_data(predicted_test, y_test, master_mask):
    return (
        get_mm_data(predicted_test, master_mask, kind="val"),
        get_mm_data(y_test.values, master_mask, kind="val"),
    )
