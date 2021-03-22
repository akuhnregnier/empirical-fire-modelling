# -*- coding: utf-8 -*-
from copy import deepcopy
from inspect import signature

import joblib
from wildfires.joblib.caching import CodeObj

from .custom_backend import Factory, HashProxy, custom_get_hash


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
    return (
        joblib.hashing.hash(signature(func)),
        joblib.hashing.hash(CodeObj(func.__code__).hashable()),
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
        processed.append(
            HashProxy(
                Factory(lambda: func(var)),
                hash_value=(
                    custom_get_hash((custom_get_hash(var), get_function_hash(func)))
                ),
            )
        )
    return tuple(processed)
