# -*- coding: utf-8 -*-

import tempfile

import joblib
import pytest

from empirical_fire_modelling import variable
from empirical_fire_modelling.cache import VariableProxyMemory


@pytest.fixture
def tmp_dir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture
def variable_proxy_memory(tmp_dir):
    return VariableProxyMemory(tmp_dir)


def test_variable_collection_get_hash(variable_proxy_memory):
    get_hash = variable_proxy_memory.get_hash

    var1 = variable.DRY_DAY_PERIOD[0]
    var2 = variable.DRY_DAY_PERIOD[9]

    assert get_hash({var1, var2}) == (
        joblib.hashing.hash(
            {
                get_hash(var1),
                get_hash(var2),
            }
        )
    )

    assert get_hash((var1, var2)) == (
        joblib.hashing.hash(
            (
                get_hash(var1),
                get_hash(var2),
            )
        )
    )

    assert get_hash((var1, var2)) == get_hash([var1, var2])
