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
from copy import deepcopy
from functools import partial, reduce, wraps
from inspect import signature
from operator import add, attrgetter

import joblib
from wildfires.data import get_memory
from wildfires.joblib.caching import CodeObj

from ..exceptions import NotCachedError
from .custom_backend import custom_get_hash, register_backend
from .same_call import extract_uniform_args_kwargs

logger = logging.getLogger(__name__)


__all__ = (
    "IN_STORE",
    "cache",
    "custom_get_hash",
    "mark_dependency",
)

# Sentinel value used to denote calls that are already cached.
IN_STORE = object()

register_backend()
_memory = get_memory("empirical_fire_modelling", backend="custom", verbose=2)


def checkattr(name):
    """Check if given attributes exist.

    Allows use of dot notation (e.g. name='a.b') due to the use of `operator.attrgetter`

    """

    def check(obj):
        try:
            attrgetter(name)(obj)
            return True
        except AttributeError:
            return False

    return check


def default_attrgetter(*args, default=None):
    """`operator.attrgetter` with a default value."""

    def retrieve(obj):
        try:
            return attrgetter(*args)(obj)
        except AttributeError:
            return default

    return retrieve


def _calculate_dependency_hash(dependencies):
    """Calculate a hash for the dependencies."""
    dependency_hashes = []

    for f in dependencies:
        for flag_check, retrieve_func in (
            # TODO: Make this more robust than simply relying on this ordering.
            # The ordering here is very important since functools.wraps() will
            # copy the '_dependency' flag and so we have to start looking at the
            # deepest possible nesting level.
            (
                checkattr("_orig_func._dependency"),
                attrgetter("_orig_func"),
            ),
            (checkattr("_dependency"), lambda f: f),
        ):
            if flag_check(f):
                func = retrieve_func(f)
                break
        else:
            raise ValueError("All dependencies must be marked with '_dependency'.")

        # Ensure that the hash can be calculated, i.e. that there are no mutable
        # objects present in the default arguments. Copy the object since some
        # object (e.g. immutabledict) will cache the hash resulting from calling
        # `hash()` (e.g. in a '_hash' attribute), and since the output of Python's
        # `hash()` function is not constant across sessions, this causes Joblib's
        # hash to change as well (which we do not want).
        hash(deepcopy(signature(func)))
        # Finally, calculate the hash using Joblib because the inbuilt hash()
        # function changes its output in between runs.
        dependency_hashes.append(joblib.hashing.hash(signature(func)))
        dependency_hashes.append(joblib.hashing.hash(CodeObj(func.__code__).hashable()))

    return joblib.hashing.hash(dependency_hashes)


def mark_dependency(f):
    """Decorator which marks a function as a potential dependency.

    Args:
        f (callable): The dependency to be recorded.

    """
    f._dependency = True
    return f


def _get_hashed(func, *args, dependencies=(), hash_func=custom_get_hash, **kwargs):
    """Calculate a hash for the call, including dependencies."""
    args, kwargs = extract_uniform_args_kwargs(func, *args, **kwargs)

    # Go through the original arguments and hash the contents manually.
    args_hashes = []
    for arg in args:
        args_hashes.append(hash_func(arg))

    # Repeat the above process for the kwargs. The keys should never include
    # MaskedArray data so we only need to deal with the values.
    kwargs_hashes = {}
    for key, arg in kwargs.items():
        kwargs_hashes[key] = hash_func(arg)

    # Hash the original function to differentiate different functions apart.
    func_code = CodeObj(func.__code__).hashable()

    return dict(
        func_code=func_code,
        args_hashes=args_hashes,
        kwargs_hashes=kwargs_hashes,
        dependencies=_calculate_dependency_hash(dependencies),
    )


def cache(*args, memory=_memory, dependencies=(), hash_func=custom_get_hash):
    """A cached function with limited MaskedArray support.

    The added method `check_in_store()` will be added and may be used to determine
    whether a given call is already cached. This method will return `IN_STORE` if the
    call is already cached, or raise a `NotCachedError` otherwise.

    Args:
        args (callable): Function to be cached.
        memory (joblib.memory.Memory): Cache instance defining the Joblib Memory
            instance to use for caching. Can be overriden, e.g. for testing.
        dependencies (tuple of callable): Other functions the cached function depends
            on. If any of these functions change from one run to the next, the cache
            will be invalidated.
        hash_func (callable): Function used to calculate hash values of function
            arguments.

    Returns:
        callable: The cached function with added `check_in_store()` method.

    """
    if not args:
        return partial(
            cache,
            memory=memory,
            dependencies=dependencies,
            hash_func=hash_func,
        )

    assert len(args) == 1
    assert callable(args[0])
    func = args[0]

    # Update dependencies to enable chaining of dependencies.
    dependencies = (
        *dependencies,
        *reduce(
            add,
            map(
                tuple,
                map(
                    default_attrgetter("_orig_func._dependencies", default=()),
                    dependencies,
                ),
            ),
            (),
        ),
    )

    def inner(hashed, args, kwargs):
        return func(*args, **kwargs)

    cached_inner = memory.cache(ignore=["args", "kwargs"])(inner)

    def bound_get_hashed(*orig_args, **orig_kwargs):
        return _get_hashed(
            func,
            *orig_args,
            dependencies=dependencies,
            hash_func=hash_func,
            **orig_kwargs,
        )

    @wraps(func)
    def cached_func(*orig_args, **orig_kwargs):
        hashed = bound_get_hashed(*orig_args, **orig_kwargs)
        return cached_inner(hashed, orig_args, orig_kwargs)

    def check_in_store(*args, **kwargs):
        """Check whether a given call to the cached function is already cached.

        Args:
            args, kwargs: Arguments to check.

        Returns:
            IN_STORE: If the given call was found in the cache.

        Raises:
            NotCachedError: If the given call was not found in the cache.

        """
        if not cached_inner.store_backend.contains_item(
            cached_inner._get_output_identifiers(
                bound_get_hashed(*args, **kwargs), args, kwargs
            )
        ):
            raise NotCachedError("The given call is not cached.")
        return IN_STORE

    cached_func.check_in_store = check_in_store
    cached_func._orig_func = func
    cached_func._orig_func._dependencies = dependencies

    return cached_func
