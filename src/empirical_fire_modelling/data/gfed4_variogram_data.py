# -*- coding: utf-8 -*-
import numpy as np
from wildfires.data import GFEDv4
from wildfires.utils import get_land_mask

from empirical_fire_modelling.cache import cache


@cache
def get_gfed4_variogram_data(i):
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

    assert inds.shape[0] == valid_indices.shape[0], "All samples should have been used"

    # print(f"Max N:    {valid_indices.shape[0]:>10d}")
    # print(f"Chosen N: {inds.shape[0]:>10d}")

    return coords[inds], ba_flat.data[inds], title
