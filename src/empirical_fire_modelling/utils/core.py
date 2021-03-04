# -*- coding: utf-8 -*-
"""Miscellaneous functions."""
import re

import numpy as np
from sklearn.model_selection import train_test_split

from ..configuration import (
    experiment_name_dict,
    fill_name,
    no_fill_feature_order,
    train_test_split_kwargs,
    units,
)

__all__ = (
    "add_units",
    "filter_by_month",
    "get_lag",
    "get_lags",
    "get_mm_data",
    "get_mm_indices",
    "repl_fill_name",
    "repl_fill_names",
    "repl_fill_names_columns",
    "sort_experiments",
    "sort_features",
    "transform_series_sum_norm",
)


def add_units(variables):
    """Add units to variables based on the `units` dict."""
    if isinstance(variables, str):
        return add_units([variables])[0]
    var_units = []
    for var in variables:
        matched_unit_vars = [
            unit_var for unit_var in units if re.search(unit_var, var) is not None
        ]
        assert (
            len(matched_unit_vars) == 1
        ), f"There should only be exactly 1 matching variable for '{var}'."
        var_units.append(f"{var} ({units[matched_unit_vars[0]]})")
    return var_units


def repl_fill_name(name, sub=""):
    fill_ins = fill_name("")
    return name.replace(fill_ins, sub)


def repl_fill_names(names, sub=""):
    if isinstance(names, str):
        return repl_fill_names((names,), sub=sub)[0]
    return [repl_fill_name(name, sub=sub) for name in names]


def repl_fill_names_columns(df, inplace=False, sub=""):
    return df.rename(
        columns=dict(
            (orig, short)
            for orig, short in zip(df.columns, repl_fill_names(df.columns, sub=sub))
        ),
        inplace=inplace,
    )


def get_lag(feature, target_feature=None):
    """Return the lag duration as an integer.
    Optionally a specific target feature can be required.
    Args:
        feature (str): Feature to extract month from.
        target_feature (str): If given, this feature is required for a successful
            match.
    Returns:
        int or None: For successful matches (see `target_feature`), an int
            representing the lag duration is returned. Otherwise, `None` is returned.
    """
    if target_feature is None:
        target_feature = ".*?"
    else:
        target_feature = re.escape(target_feature)

    # Avoid dealing with the fill naming.
    feature = repl_fill_name(feature)

    match = re.search(target_feature + r"\s-(\d+)\s", feature)

    if match is None:
        # Try matching to 'short names'.
        match = re.search(target_feature + r"(\d+)M", feature)

    if match is not None:
        return int(match.groups(default="0")[0])
    if match is None and re.match(target_feature, feature):
        return 0
    return None


def get_lags(features, target_feature=None):
    if not isinstance(features, str):
        return [get_lag(feature, target_feature=target_feature) for feature in features]
    return get_lag(features, target_feature=target_feature)


def filter_by_month(features, target_feature, max_month):
    """Filter feature names using a single target feature and maximum month.
    Args:
        features (iterable of str): Feature names to select from.
        target_feature (str): String in `features` to match against.
        max_month (int): Maximum month.
    Returns:
        iterable of str: The filtered feature names, subset of `features`.
    """
    filtered = []
    for feature in features:
        lag = get_lag(feature, target_feature=target_feature)
        if lag is not None and lag <= max_month:
            filtered.append(feature)
    return filtered


def sort_experiments(experiments):
    """Sort experiments based on `experiment_name_dict`."""
    name_lists = (
        list(experiment_name_dict.keys()),
        list(experiment_name_dict.values()),
    )
    order = []
    experiments = list(experiments)
    for experiment in experiments:
        for name_list in name_lists:
            if experiment in name_list:
                order.append(name_list.index(experiment))
                break
        else:
            # No break encountered, so no order could be found.
            raise ValueError(f"Experiment {experiment} could not be found.")
    out = []
    for i in np.argsort(order):
        out.append(experiments[i])
    return out


def sort_features(features):
    """Sort feature names using their names and shift magnitudes.
    Args:
        features (iterable of str): Feature names to sort.
    Returns:
        list of str: Sorted list of features.
    """
    raw_features = []
    lags = []
    for feature in features:
        lag = get_lag(feature)
        assert lag is not None
        # Remove fill naming addition.
        feature = repl_fill_name(feature)
        if str(lag) in feature:
            # Strip lag information from the string.
            raw_features.append(feature[: feature.index(str(lag))].strip("-").strip())
        else:
            raw_features.append(feature)
        lags.append(lag)
    sort_tuples = tuple(zip(features, raw_features, lags))
    return [
        s[0]
        for s in sorted(
            sort_tuples, key=lambda x: (no_fill_feature_order[x[1]], abs(int(x[2])))
        )
    ]


def transform_series_sum_norm(x):
    x = x / np.sum(np.abs(x))
    return x


def get_mm_indices(master_mask, train_test_split_kwargs=train_test_split_kwargs):
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
