# -*- coding: utf-8 -*-
from wildfires.dask_cx1 import get_client

from ..exceptions import NotCachedError
from .core import get_model

__all__ = ("call_get_model_check_cache",)


def call_get_model_check_cache(X_train, y_train, param_dict, cache_check=False):
    """Get the model using `get_model()`, but only create the Dask client if needed.

    Args:
        X_train, y_train, param_dict: See `get_model()`.
        cache_check (bool): If True, verify that the call is already cached.

    Raises:
        NotCachedError: If `cache_check` and the call to `get_model()` is not already
            cached.

    Returns:
        rf, client: Random forest instance and the Dask client. Note that the Dask
            client is None is it is not needed.

    """
    client = None
    cached_model = True
    try:
        get_model(X_train, y_train, param_dict, cache_check=True)
    except NotCachedError:
        cached_model = False

    if not cached_model:
        if cache_check:
            raise NotCachedError("Model is not cached.")
        else:
            # Get Dask client that is needed for model fitting.
            client = get_client(fallback=True, fallback_threaded=True)

    rf = get_model(X_train, y_train, param_dict)
    return rf, client
