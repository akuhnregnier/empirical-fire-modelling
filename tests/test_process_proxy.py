# -*- coding: utf-8 -*-
from empirical_fire_modelling.cache import custom_get_hash, process_proxy

from .utils import *  # noqa


def test_lazy_cache_checking(test_cache):
    """Proxy objects should not be realised when checking cache contents.

    This should also apply to processed proxies.

    """

    @test_cache
    def f(x):
        return x + 1

    def process_func(y):
        return y + 10

    # Generate cache entries.
    assert process_proxy((f(1),), (process_func,)) == (12,)

    # Retrieve the proxy pointing to the cached result.
    f1_proxy = f(1)
    assert process_proxy((f1_proxy,), (process_func,)) == (12,)

    # The original proxy should have been called above because `process_func` does not
    # do any caching.
    assert f1_proxy.__factory__.was_called

    # However, the hash of the `process_proxy` result should not depend on whether the
    # input HashProxy objects have been called or not.

    # Retrieve the proxy pointing to the cached result.
    f1_proxy = f(1)
    lazy_hash = custom_get_hash(process_proxy((f1_proxy,), (process_func,))[0])
    assert not f1_proxy.__factory__.was_called

    # Realise the proxy.
    str(f1_proxy)
    real_hash = custom_get_hash(process_proxy((f1_proxy,), (process_func,))[0])
    assert f1_proxy.__factory__.was_called

    # The two hash values should match.
    assert lazy_hash == real_hash

    # Ensure proxies are realised.
    f1_proxy = f(1)
    processed = process_proxy((f1_proxy,), (process_func,))[0]
    assert not f1_proxy.__factory__.was_called
    assert not processed.__factory__.was_called

    # Realise both proxies.
    assert processed == 12
    assert f1_proxy.__factory__.was_called
    assert processed.__factory__.was_called
