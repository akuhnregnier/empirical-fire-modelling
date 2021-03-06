# -*- coding: utf-8 -*-


from empirical_fire_modelling.cache import check_in_store

from .utils import *  # noqa


def test_same_signature(test_cache):
    """The way a cached function is called should not affect the caching behaviour."""

    @test_cache
    def f(x, y=10):
        return x + y

    assert f(1) == 11
    assert check_in_store(f, 1)
    assert check_in_store(f, 1, y=10)
    assert check_in_store(f, 1, 10)
    assert check_in_store(f, x=1, y=10)
    assert check_in_store(f, y=10, x=1)


def test_same_signature_kwargs(test_cache):
    """The way a cached function is called should not affect the caching behaviour."""

    @test_cache
    def f(x, y=10, **kwargs):
        return x + y + sum(map(ord, kwargs.keys())) + sum(kwargs.values())

    assert f(1, a=10) == 118
    assert check_in_store(f, 1, y=10, a=10)
    assert check_in_store(f, 1, a=10, y=10)
    assert check_in_store(f, 1, 10, a=10)


def test_same_signature_args_kwargs(test_cache):
    """The way a cached function is called should not affect the caching behaviour."""

    @test_cache
    def f(x, *my_args, y=4, **my_kwargs):
        return (
            x
            + sum(my_args)
            + y
            + sum(map(ord, my_kwargs.keys()))
            + sum(my_kwargs.values())
        )

    assert f(1, 2, 3, y=4, a=10, b=20) == 235
    assert check_in_store(f, 1, 2, 3, a=10, b=20)
