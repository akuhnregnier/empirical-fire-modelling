# -*- coding: utf-8 -*-
"""Miscellaneous functions."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..configuration import train_test_split_kwargs

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


__all__ = (
    "get_mm_data",
    "get_mm_indices",
    "tqdm",
    "transform_series_sum_norm",
)


def transform_series_sum_norm(x):
    """Transform a DataFrame using the sum of absolute values."""
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Expected pandas.DataFrame, got '{x}'.")
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
