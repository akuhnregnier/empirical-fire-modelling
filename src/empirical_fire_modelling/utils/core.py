# -*- coding: utf-8 -*-
"""Miscellaneous functions."""

import os
from copy import copy

import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from wildfires.dask_cx1 import get_client as wildfires_get_client
from wildfires.qstat import get_ncpus

from ..configuration import train_test_split_kwargs
from ..exceptions import NotCachedError

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


__all__ = (
    "get_client",
    "get_mm_data",
    "get_mm_indices",
    "optional_client_call",
    "tqdm",
    "transform_series_sum_norm",
)


def transform_series_sum_norm(x):
    """Transform a DataFrame using the sum of absolute values."""
    if not isinstance(x, pd.Series):
        raise ValueError(f"Expected pandas.Series, got '{x}'.")
    x = x / np.sum(np.abs(x))
    return x


def get_mm_indices(master_mask, train_test_split_kwargs=train_test_split_kwargs):
    """Return flattened train and test indices into the arrays."""
    mm_valid_indices = np.where(~master_mask.ravel())[0]
    mm_valid_train_indices, mm_valid_val_indices = train_test_split(
        mm_valid_indices,
        **train_test_split_kwargs,
    )
    return mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices


def get_mm_data(x, master_mask, kind):
    """Return masked master_mask copy and training or validation indices.

    The master_mask copy is filled using the given data.

    Args:
        x (array-like): Data to use.
        master_mask (array):
        kind ({'train', 'val'})

    Returns:
        masked_data, mm_indices:

    """
    mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices = get_mm_indices(
        master_mask
    )
    masked_data = np.ma.MaskedArray(
        np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
    )
    if kind == "train":
        masked_data.ravel()[mm_valid_train_indices] = x
    elif kind == "val":
        masked_data.ravel()[mm_valid_val_indices] = x
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return masked_data


def get_client(*args, **kwargs):
    """Wrapper around wildfires.dask_cx1.get_client.

    Only tries to connect to a distributed scheduler if not running as a CX1 job. This
    is controlled by an environment variable.

    """
    if "RUNNING_AS_JOB" in os.environ:
        # Do not connect to a distributed scheduler.
        return Client(n_workers=1, threads_per_worker=get_ncpus())
    else:
        return wildfires_get_client(*args, **kwargs)


def optional_client_call(func, call_kwargs, cache_check=False, add_client=False):
    """Call `func` and only create a Dask client if needed.

    Args:
        func(callable): Function to call.
        call_kwargs: Function call keyword arguments.
        cache_check (bool): If True, only verify that the call is already cached.
        add_client (bool): If True, add the created Client to `call_kwargs` before
            calling `func`. `call_kwargs` will be copied using `copy.copy()` prior to
            this. This will be done even if the call is already cached, in which case
            `client=None` will be added to `call_kwargs`.

    Raises:
        NotCachedError: If `cache_check` and the call to `func(**call_kwargs)` is not
            already cached.

    Returns:
        result, client: Output of `func(**call_kwargs)` and the Dask client. Note that
            the Dask client is None is it is not needed.

    """
    client = None
    cached_result = True

    if add_client:
        # Add None (for now), since cached functions must ignore this argument.
        # Therefore its value shouldn't affect cache checking.
        call_kwargs = copy(call_kwargs)
        call_kwargs["client"] = client

    if hasattr(func, "check_in_store"):
        # The function is cached.
        try:
            func.check_in_store(**call_kwargs)
        except NotCachedError:
            cached_result = False
    else:
        try:
            func(cache_check=True, **call_kwargs)
        except NotCachedError:
            cached_result = False

    if not cached_result:
        if cache_check:
            raise NotCachedError("Result is not cached.")
        else:
            # Get Dask client that is needed for model fitting.
            client = get_client(fallback=True, fallback_threaded=True)
            if add_client:
                call_kwargs["client"] = client

    return func(**call_kwargs), client
