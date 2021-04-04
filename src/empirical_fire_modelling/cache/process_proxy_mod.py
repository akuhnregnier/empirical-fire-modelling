# -*- coding: utf-8 -*-
import pickle
import traceback
from copy import deepcopy
from inspect import signature
from pathlib import Path

import joblib
from joblib._store_backends import concurrency_safe_rename, concurrency_safe_write
from wildfires.joblib.caching import CodeObj

from .core import _memory
from .custom_backend import Factory, HashProxy, custom_get_hash

MAP_FILENAME = Path(_memory.location) / "joblib" / "_process_proxy"


def _concurrency_safe_write(to_write, filename, write_func):
    """Writes an object into a file in a concurrency-safe way."""
    try:
        temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    except:
        print("Something went wrong before moving the file.!")
        traceback.print_exc()
    concurrency_safe_rename(temporary_filename, filename)


def load_hash_map():
    if not MAP_FILENAME.is_file():
        return {}
    with MAP_FILENAME.open("rb") as f:
        return pickle.load(f)


def get_processed_hash(key, func, var):
    if not hasattr(get_processed_hash, "hash_map"):
        get_processed_hash.hash_map = load_hash_map()
    else:
        if key in get_processed_hash.hash_map:
            return get_processed_hash.hash_map[key]

        # Update the mapping before giving up and calculating the result.
        get_processed_hash.hash_map = load_hash_map()

    if key in get_processed_hash.hash_map:
        return get_processed_hash.hash_map[key]

    # Key was not found, so we have to evaluate the function in order to determine the
    # resulting hash.
    processed_hash = custom_get_hash(func(var))

    # Reload the map in case it changed in the meantime.
    get_processed_hash.hash_map = load_hash_map()
    # Add the new hash to the map.
    # NOTE - this will leave a small window of time where concurrent access may result
    # in new hashes to not be recorded.
    get_processed_hash.hash_map[key] = processed_hash

    # Save the updated mapping.

    def write_func(to_write, dest_filename):
        with open(dest_filename, "wb") as f:
            pickle.dump(to_write, f, protocol=-1)

    _concurrency_safe_write(get_processed_hash.hash_map, MAP_FILENAME, write_func)


def get_function_hash(func):
    # Ensure that the hash can be calculated, i.e. that there are no mutable
    # objects present in the default arguments. Copy the object since some
    # object (e.g. immutabledict) will cache the hash resulting from calling
    # `hash()` (e.g. in a '_hash' attribute), and since the output of Python's
    # `hash()` function is not constant across sessions, this causes Joblib's
    # hash to change as well (which we do not want).
    hash(deepcopy(signature(func)))
    # Finally, calculate the hash using Joblib because the inbuilt hash()
    # function changes its output in between runs.
    return joblib.hashing.hash(signature(func)) + joblib.hashing.hash(
        CodeObj(func.__code__).hashable()
    )


def process_proxy(output, functions):
    """Lazily apply deterministic `functions` to `output`.

    Note that the functions should not contain closure variables, since these will not
    (currently) influence the functions hash value.

    Args:
        output (N-tuple): Tuple of return values. May consist of `HashProxy` instances.
        functions (N-tuple of callable): Deterministic functions to apply to `output`.
            The i-th function will be applied to the i-th output.

    Returns:
        tuple of HashProxy: Lazily applied functions onto output.

    Raises:
        ValueError: If `len(output) != len(functions)`.

    """
    if len(output) != len(functions):
        raise ValueError(
            "Expected the same number of outputs and functions. "
            f"Got {len(output)} and {len(functions)}."
        )

    processed = []
    for var, func in zip(output, functions):
        lazy_hash_key = custom_get_hash(var) + get_function_hash(func)
        processed_hash = get_processed_hash(lazy_hash_key, func, var)

        processed.append(
            HashProxy(
                Factory(lambda: func(var)),
                hash_value=processed_hash,
            )
        )
    return tuple(processed)
