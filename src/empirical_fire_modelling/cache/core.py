# -*- coding: utf-8 -*-
"""Caching of results.

Note that dataframe column name changes may not trigger recalculations.

Due to the custom implementation of lazy proxied objects being returned from the
custom Joblib backend, all cached functions within this module should be decorated
exclusively with the `cache` decorator defined here. To minimise the possibility of a
Proxy object being realised, as many functions as possible should be cached (at the
expense of storage, of course).

Calling repr() on Proxy objects is fine, but calling str() will realise them
(i.e. call the factory function), so e.g. bare print() statements should not be used.

"""
import logging
from functools import wraps

from wildfires.data import get_memory, ma_cache
from wildfires.joblib.caching import wrap_decorator

from ..exceptions import InvalidCacheCheck, NotCachedError
from .custom_backend import custom_get_hash, register_backend

logger = logging.getLogger(__name__)


__all__ = ("IN_STORE", "cache", "check_in_store", "custom_get_hash")

# Sentinel value used to denote calls that are already cached.
IN_STORE = object()

register_backend()
_memory = get_memory("empirical_fire_modelling", backend="custom", verbose=2)
_cache = ma_cache(memory=_memory, hash_func=custom_get_hash)


# Add presence-checking functionality.
@wrap_decorator
def cache(func, ma_cache_inst=_cache):
    """A cached function with limited MaskedArray support.

    The keyword argument `cache_check` will be added and used automatically to
    determine whether a given call is already cached, if requested by the caller by
    giving `cache_check=True`. In this case, the cached function will return
    `IN_STORE` if the call is already cached, or raise a `NotCachedError` otherwise.

    Args:
        func (callable): Function to be cached.
        ma_cache_inst (wildfires.data.ma_cache): Cache instance defining the Joblib
            Memory instance to use for caching. Can be overriden e.g. for testing.

    """
    cached_func = ma_cache_inst(func)

    @wraps(func)
    def cached_check(*args, cache_check=False, **kwargs):
        if cache_check:
            if not cached_func.is_cached(*args, **kwargs):
                raise NotCachedError("The given call is not cached.")
            return IN_STORE
        # Otherwise continue on as normal.
        return cached_func(*args, **kwargs)

    return cached_check


def check_in_store(cached_func, *args, **kwargs):
    """Convenience function to check whether a given call is already cached."""
    if cached_func(*args, cache_check=True, **kwargs) is not IN_STORE:
        raise InvalidCacheCheck(
            f"The function {cached_func} is misconfigured for cache checking."
        )
    return IN_STORE
