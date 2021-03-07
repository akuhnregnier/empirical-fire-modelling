# -*- coding: utf-8 -*-

import numpy as np
import pytest

from empirical_fire_modelling.cache import IN_STORE, custom_get_hash
from empirical_fire_modelling.exceptions import NotCachedError

from .utils import *  # noqa


def test_uncached_exception(test_cache):
    @test_cache
    def f(x):
        """Function to be cached."""
        return x + 1

    with pytest.raises(NotCachedError):
        f.check_in_store(0)

    # The previous call should not have run the function and therefore there should
    # still be no cached output.
    with pytest.raises(NotCachedError):
        f.check_in_store(0)


def test_cache_checking(test_cache):
    @test_cache
    def f(x):
        """Function to be cached."""
        return x + 1

    with pytest.raises(NotCachedError):
        f.check_in_store(0)

    # Call the function to generate the cache entry.
    assert f(0) == 1
    # There should no longer be an error when checking the call.
    assert f.check_in_store(0) is IN_STORE


def test_chained_cache_checking(test_cache):
    """Test cache checking for chained cached functions."""

    @test_cache
    def inner(x):
        return x + 1

    @test_cache
    def outer(x):
        return inner(x) + 1

    for f in (outer, inner):
        with pytest.raises(NotCachedError):
            f.check_in_store(0)

    # Call the inner function to generate the inner cache entry.
    assert inner(0) == 1
    # There should no longer be an error when checking the inner call.
    assert inner.check_in_store(0) is IN_STORE

    # The outer call should still be uncached.
    with pytest.raises(NotCachedError):
        outer.check_in_store(0)

    # Call the outer function to generate its cache entry.
    assert outer(0) == 2

    # There should no longer be an error when checking the call.
    assert outer.check_in_store(0) is IN_STORE


def test_chained_mixed_cache_checking(test_cache):
    """Test cache checking for chained functions where only some are cached."""

    @test_cache
    def inner(x):
        return x + 1

    def outer(x, cache_check=False):
        if cache_check:
            # Ensure both calls to the 'expensive' cached function are cached
            # properly.
            inner.check_in_store(x)
            return inner.check_in_store(x + 1)
        return inner(x) + inner(x + 1) + 1

    with pytest.raises(NotCachedError):
        inner.check_in_store(0)
    with pytest.raises(NotCachedError):
        outer(0, cache_check=True)

    # Call the outer function to generate the inner cache entry.
    assert outer(0) == 4

    # There should no longer be an error when checking the calls.
    assert inner.check_in_store(0) is IN_STORE
    assert outer(0, cache_check=True) is IN_STORE


def test_ma_cache(test_cache, dummy_dataset):
    @test_cache
    def dummy_func(x, *args, **kwargs):
        return x

    dummy_dataset.cube.data.mask[5, 180, 300] = True

    args = (
        (1, 2, 3),
        2,
        3,
    )
    kwargs = dict(
        a=10,
        b=20,
        c=np.ma.MaskedArray([1, 2, 3], mask=[1, 0, 1]),
    )

    with pytest.raises(NotCachedError):
        dummy_func.check_in_store(dummy_dataset, *args, **kwargs)

    assert dummy_dataset == dummy_func(dummy_dataset, *args, **kwargs)

    assert dummy_func.check_in_store(dummy_dataset, *args, **kwargs)

    # Test the loading too.
    loaded = dummy_func(dummy_dataset, *args, **kwargs)
    assert loaded is not dummy_dataset
    assert loaded.cube.data is not dummy_dataset.cube.data
    assert loaded == dummy_dataset
    assert custom_get_hash(loaded.cube.data) == custom_get_hash(dummy_dataset.cube.data)


def test_multiple_function_cache(test_cache):
    """Ensure that the decorator can discern between different functions."""

    @test_cache
    def dummy_func1(x):
        return x + 1

    @test_cache
    def dummy_func2(x):
        return x + 2

    with pytest.raises(NotCachedError):
        dummy_func1.check_in_store(0)
    with pytest.raises(NotCachedError):
        dummy_func2.check_in_store(0)

    assert dummy_func1(0) == 1
    assert dummy_func2(0) == 2

    assert dummy_func1.check_in_store(0)
    assert dummy_func2.check_in_store(0)

    # Now test the previously cached versions.
    assert dummy_func1(0) == 1
    assert dummy_func2(0) == 2
