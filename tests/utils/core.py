# -*- coding: utf-8 -*-
import tempfile

import iris
import numpy as np
import pytest
from joblib import Memory
from wildfires.data import Datasets, MonthlyDataset, dummy_lat_lon_cube

from empirical_fire_modelling.cache import cache


@pytest.fixture
def tmp_dir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture
def dummy_memory(tmp_dir):
    return Memory(tmp_dir, backend="custom")


@pytest.fixture
def test_cache(dummy_memory):
    return cache(memory=dummy_memory)


@pytest.fixture
def dummy_dataset():
    class DummyDataset(MonthlyDataset):
        def __init__(self):
            self.cubes = iris.cube.CubeList(
                [
                    dummy_lat_lon_cube(
                        np.ma.MaskedArray(
                            np.random.default_rng(0).random((10, 360, 720)),
                            mask=np.zeros((10, 360, 720), dtype=np.bool_),
                        )
                    )
                ]
            )

    return DummyDataset()


@pytest.fixture
def dummy_datasets(dummy_dataset):
    return Datasets([dummy_dataset])
