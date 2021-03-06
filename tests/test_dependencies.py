# -*- coding: utf-8 -*-

import pytest

from empirical_fire_modelling.cache import check_in_store, mark_dependency
from empirical_fire_modelling.exceptions import NotCachedError

from .utils import *  # noqa


def test_dependencies(test_cache):
    """Test that when dependencies change, the cache gets invalidated."""

    @test_cache
    @mark_dependency
    def f(x):
        return x + 1

    @test_cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    assert f(1) == 2
    assert f2(1) == 12

    # Defining the same functions as above should not invalidate the previous cache
    # entries.

    @test_cache
    @mark_dependency
    def f(x):
        return x + 1

    @test_cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    assert check_in_store(f, 1)
    assert check_in_store(f2, 1)

    # However, redefining `f` should invalidate both cache entries.

    @test_cache
    @mark_dependency
    def f(x):
        return x + 2

    @test_cache(dependencies=(f,))
    def f2(x):
        return f(x) + 10

    with pytest.raises(NotCachedError):
        check_in_store(f, 1)

    with pytest.raises(NotCachedError):
        check_in_store(f2, 1)

    assert f(1) == 3
    assert f2(1) == 13
