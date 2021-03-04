# -*- coding: utf-8 -*-
"""Caching of results.

Note that dataframe column name changes may not trigger recalculations.

"""
import logging
from functools import wraps

from wildfires.data import get_memory, ma_cache
from wildfires.joblib.caching import wrap_decorator
from wildfires.joblib.cloudpickle_backend import register_backend as register_cl_backend

from ..exceptions import InvalidCacheCheck, NotCachedError

logger = logging.getLogger(__name__)


__all__ = ("IN_STORE", "cache", "check_in_store")

# Sentinel value used to denote calls that are already cached.
IN_STORE = object()

register_cl_backend()
_memory = get_memory("empirical_fire_modelling", backend="cloudpickle", verbose=2)
_cache = ma_cache(memory=_memory)


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
