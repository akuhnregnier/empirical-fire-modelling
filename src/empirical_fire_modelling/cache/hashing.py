# -*- coding: utf-8 -*-

import joblib
import numpy as np
import pandas as pd
import xxhash
from wildfires.data import Dataset, Datasets

from .. import variable


def hash_ma(x):
    """Compute the hash for a numpy MaskedArray."""
    return xxhash.xxh64_hexdigest(x.data) + xxhash.xxh64_hexdigest(x.mask)


def hash_dataset(dataset):
    """Compute the hash of a Dataset.

    Note: This realises any lazy data.

    """
    # Compute the hash for each piece of data.
    dataset_hash = ""
    for cube in dataset:
        if isinstance(cube.data, np.ma.core.MaskedArray):
            dataset_hash += hash_ma(cube.data)
        else:
            dataset_hash += xxhash.xxh64_hexdigest(cube.data)

    # Finally consider the coordinates and metadata.
    dataset_hash += joblib.hashing.hash(dataset._shallow)

    return dataset_hash


def hash_df(df):
    """Compute the hash of a pandas DataFrame.

    This only considers the index, data, and column names.

    """
    dataset_hash = xxhash.xxh64_hexdigest(np.ascontiguousarray(df.values))
    dataset_hash += joblib.hashing.hash(df.index)
    dataset_hash += joblib.hashing.hash(df.columns)
    return dataset_hash


def get_variable_parent_name(var):
    """Get the name of the parent if it exists or return the variable name otherwise."""
    if hasattr(var, "parent"):
        return var.parent.name
    else:
        return var.name


def get_hash(arg):
    """Compute a hash with special support for e.g. MaskedArray."""
    if isinstance(arg, np.ma.core.MaskedArray):
        arg_hash = hash_ma(arg)
    elif isinstance(arg, Datasets):
        arg_hash = ""
        for dataset in arg:
            arg_hash += hash_dataset(dataset)
    elif isinstance(arg, Dataset):
        arg_hash = hash_dataset(arg)
    elif isinstance(arg, pd.DataFrame):
        arg_hash = hash_df(arg)
    # Ignore the Variable 'rank' and 'units' attributes for the hash.
    elif isinstance(arg, variable.OffsetVariable):
        arg_hash = joblib.hashing.hash(
            (arg.name, arg.shift, arg.comp_shift, get_variable_parent_name(arg))
        )
    elif isinstance(arg, variable.Variable):
        arg_hash = joblib.hashing.hash(
            (arg.name, arg.shift, get_variable_parent_name(arg))
        )
    else:
        arg_hash = joblib.hashing.hash(arg)
    return arg_hash
