# -*- coding: utf-8 -*-


from empirical_fire_modelling.cache import check_in_store

from .utils import *  # noqa


def test_lazy_cache_checking(test_cache):
    """Proxy objects should not be realised when checking cache contents."""

    @test_cache
    def f(x):
        return x + 1

    @test_cache
    def f2(x):
        return x + 10

    # Generate cache entries.
    assert f(1) == 2
    assert f2(f(1)) == 12

    # Retrieve the proxy pointing to the cached result.
    f1_proxy = f(1)
    assert check_in_store(f2, f1_proxy)
    assert f2(f1_proxy) == 12

    assert not f1_proxy.__factory__.was_called


def test_chained_lazy_cache_checking(test_cache):
    """Proxy objects should not be realised when checking cache contents."""

    @test_cache
    def input_proxy():
        return 1

    @test_cache
    def f(x):
        return x + 1

    def f2(x, cache_check=False):
        if cache_check:
            return check_in_store(f, x)
        return f(x) + 10

    # Generate cache entries.
    assert f2(input_proxy()) == 12

    # Retrieve the proxy pointing to the cached result.
    in_proxy = input_proxy()
    assert check_in_store(f2, in_proxy)
    assert f2(in_proxy) == 12

    assert not in_proxy.__factory__.was_called
