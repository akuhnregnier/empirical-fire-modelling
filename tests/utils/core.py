# -*- coding: utf-8 -*-
import tempfile

import iris
import numpy as np
import pytest
from joblib import Memory
from wildfires.data import Datasets, MonthlyDataset, dummy_lat_lon_cube, ma_cache

from empirical_fire_modelling.cache import cache, custom_get_hash


@pytest.fixture
def dummy_memory():
    tmp_dir = tempfile.TemporaryDirectory()
    yield Memory(tmp_dir.name, backend="custom")
    tmp_dir.cleanup()


@pytest.fixture
def ma_cache_inst(dummy_memory):
    return ma_cache(memory=dummy_memory, hash_func=custom_get_hash)


@pytest.fixture
def test_cache(ma_cache_inst):
    return cache(ma_cache_inst=ma_cache_inst)


@pytest.fixture
def dummy_dataset():
    class DummyDataset(MonthlyDataset):
        def __init__(self):
            self.cubes = iris.cube.CubeList(
                [dummy_lat_lon_cube(np.random.default_rng(0).random((10, 20, 40)))]
            )

    return DummyDataset()


@pytest.fixture
def dummy_datasets(dummy_dataset):
    return Datasets([dummy_dataset])
