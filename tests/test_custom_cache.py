# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from empirical_fire_modelling.cache import check_in_store, custom_get_hash
from empirical_fire_modelling.cache.custom_backend import HashProxy
from empirical_fire_modelling.exceptions import NotCachedError

from .utils import *  # noqa


@pytest.mark.parametrize(
    "value",
    [
        1,
        dummy_dataset,
        dummy_datasets,
        np.arange(10),
        np.ma.MaskedArray(np.arange(3), mask=[1, 0, 0]),
        pd.DataFrame([np.arange(10)]),
    ],
)
def test_custom_backend(test_cache, value):
    @test_cache
    def f(x):
        """Function to cache."""
        return value

    with pytest.raises(NotCachedError):
        check_in_store(f, 0)

    assert not isinstance(f(0), HashProxy)

    out = f(0)
    assert isinstance(out, HashProxy)

    cached_hash = out.hashed_value
    assert not out.__factory__.was_called

    str(out)
    assert custom_get_hash(out) == cached_hash
    assert out.__factory__.was_called
    assert isinstance(out, type(value))


@pytest.mark.parametrize(
    "input_arg",
    [
        1,
        dummy_dataset,
        dummy_datasets,
        np.arange(10),
        np.ma.MaskedArray(np.arange(3), mask=[1, 0, 0]),
        pd.DataFrame([np.arange(10)]),
    ],
)
@pytest.mark.parametrize(
    "value1",
    [
        1,
        dummy_dataset,
        dummy_datasets,
        np.arange(10),
        np.ma.MaskedArray(np.arange(3), mask=[1, 0, 0]),
        pd.DataFrame([np.arange(10)]),
    ],
)
@pytest.mark.parametrize(
    "value2",
    [
        1,
        dummy_dataset,
        dummy_datasets,
        np.arange(10),
        np.ma.MaskedArray(np.arange(3), mask=[1, 0, 0]),
        pd.DataFrame([np.arange(10)]),
    ],
)
def test_chained_factory_not_called(test_cache, input_arg, value1, value2):
    @test_cache
    def func1(x):
        """Function to cache."""
        return value1

    @test_cache
    def func2(x):
        """Function to cache."""
        return value2

    with pytest.raises(NotCachedError):
        check_in_store(func1, input_arg)

    with pytest.raises(NotCachedError):
        check_in_store(func2, value1)

    # Generate the cache entries.
    assert np.all(func1(input_arg) == value1)
    assert np.all(func2(value1) == value2)

    # Load proxies.
    out1 = func1(input_arg)
    out2 = func2(value1)
    assert isinstance(out1, HashProxy)
    assert isinstance(out2, HashProxy)

    for out in (out1, out2):
        assert not out.__factory__.was_called

    # Generate new cache entry.
    assert np.all(func2(value1) == value2)

    # Try to retrieve the cached entry from func2 using the Proxy object returned by
    # func1. This should not trigger the Proxy factory function.
    lazy_out = func2(out1)

    for out in (lazy_out, out1):
        assert isinstance(out, HashProxy)
        assert not out.__factory__.was_called

    for out, expected in ((lazy_out, value2), (out1, value1)):
        assert np.all(out == expected)
        assert out.__factory__.was_called
