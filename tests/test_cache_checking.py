# -*- coding: utf-8 -*-
import tempfile

import pytest
from joblib import Memory
from wildfires.data import ma_cache

from empirical_fire_modelling.cache import IN_STORE, cache, check_in_store
from empirical_fire_modelling.exceptions import NotCachedError


@pytest.fixture
def dummy_memory():
    tmp_dir = tempfile.TemporaryDirectory()
    yield Memory(tmp_dir.name)
    tmp_dir.cleanup()


@pytest.fixture
def ma_cache_inst(dummy_memory):
    return ma_cache(memory=dummy_memory)


@pytest.fixture()
def test_cache(ma_cache_inst):
    return cache(ma_cache_inst=ma_cache_inst)


def test_uncached_exception(test_cache):
    @test_cache
    def f(x):
        """Function to be cached."""
        return x + 1

    with pytest.raises(NotCachedError):
        check_in_store(f, 0)

    # The previous call should not have run the function and therefore there should
    # still be no cached output.
    with pytest.raises(NotCachedError):
        check_in_store(f, 0)


def test_cache_checking(test_cache):
    @test_cache
    def f(x):
        """Function to be cached."""
        return x + 1

    with pytest.raises(NotCachedError):
        check_in_store(f, 0)

    # Call the function to generate the cache entry.
    assert f(0) == 1
    # There should no longer be an error when checking the call.
    assert check_in_store(f, 0) is IN_STORE


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
            check_in_store(f, 0)

    # Call the inner function to generate the inner cache entry.
    assert inner(0) == 1
    # There should no longer be an error when checking the inner call.
    assert check_in_store(inner, 0) is IN_STORE

    # The outer call should still be uncached.
    with pytest.raises(NotCachedError):
        check_in_store(outer, 0)

    # Call the outer function to generate its cache entry.
    assert outer(0) == 2

    # There should no longer be an error when checking the call.
    assert check_in_store(outer, 0) is IN_STORE


def test_chained_mixed_cache_checking(test_cache):
    """Test cache checking for chained functions where only some are cached."""

    @test_cache
    def inner(x):
        return x + 1

    def outer(x, cache_check=False):
        if cache_check:
            # Ensure both calls to the 'expensive' cached function are cached
            # properly.
            check_in_store(inner, x)
            return check_in_store(inner, x + 1)
        return inner(x) + inner(x + 1) + 1

    for f in (outer, inner):
        with pytest.raises(NotCachedError):
            check_in_store(f, 0)

    # Call the outer function to generate the inner cache entry.
    assert outer(0) == 4

    # There should no longer be an error when checking the calls.
    for f in (outer, inner):
        assert check_in_store(f, 0) is IN_STORE
