# -*- coding: utf-8 -*-

import numpy as np

from empirical_fire_modelling.cache.hashing import get_hash

from .utils import *  # noqa


def test_ndarray_get_hash():
    data = np.arange(int(1e5))
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


def test_maskedarray_get_hash():
    data = np.ma.MaskedArray(np.arange(int(1e5)))
    data.mask = data.data > int(1e4)
    orig_hash = get_hash(data)
    data[int(1e4)] = 0
    assert get_hash(data) != orig_hash


def test_dataset_get_hash(dummy_dataset):
    # Test that the hash changes when data is changed.
    orig_hash = get_hash(dummy_dataset)
    dummy_dataset.cube.data.data[5, 180, 360] = 0
    mod_hash = get_hash(dummy_dataset)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    dummy_dataset.cube.data.mask[5, 180, 360] = True
    mask_mod_hash = get_hash(dummy_dataset)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    dummy_dataset.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(dummy_dataset)
    assert coord_mod_hash != mask_mod_hash


def test_datasets_get_hash(dummy_datasets):
    # Test that the hash changes when data is changed.
    orig_hash = get_hash(dummy_datasets)
    dummy_datasets.cube.data.data[5, 180, 360] = 0
    mod_hash = get_hash(dummy_datasets)
    assert mod_hash != orig_hash

    # Test that the hash changes if the mask changes.
    dummy_datasets.cube.data.mask[5, 180, 360] = True
    mask_mod_hash = get_hash(dummy_datasets)
    assert mask_mod_hash != mod_hash

    # Test that the hash changes if a coordinate is altered.
    dummy_datasets.cube.coord("time").long_name = "testing"
    coord_mod_hash = get_hash(dummy_datasets)
    assert coord_mod_hash != mask_mod_hash
