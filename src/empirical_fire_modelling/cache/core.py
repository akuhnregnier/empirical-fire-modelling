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
from copy import copy
from functools import partial, reduce, wraps
from inspect import signature
from operator import add, attrgetter

import joblib
from wildfires.data import get_memory
from wildfires.data.ma_cache import Decorated
from wildfires.joblib.caching import CodeObj

from ..exceptions import InvalidCacheCheck, NotCachedError
from .custom_backend import custom_get_hash, register_backend
from .same_call import extract_uniform_args_kwargs

logger = logging.getLogger(__name__)


__all__ = (
    "DepMACache",
    "IN_STORE",
    "cache",
    "check_in_store",
    "custom_get_hash",
    "mark_dependency",
)

# Sentinel value used to denote calls that are already cached.
IN_STORE = object()

register_backend()
_memory = get_memory("empirical_fire_modelling", backend="custom", verbose=2)


class DepMACache:
    """MaskedArray-capable Joblib Memory decorator.

    This is achieved by looking for MaskedArray instances in certain predefined
    locations (e.g. within any Dataset object in `args`) and calculating the hash of
    the data and mask separately before telling Joblib to ignore this object.

    """

    def __init__(self, *, memory=None, hash_func=custom_get_hash):
        if memory is None:
            raise ValueError("A Joblib.memory.Memory instance must be given.")
        self.memory = memory
        self.hash_func = hash_func

    @staticmethod
    def calculate_dependency_hash(dependencies):
        """Calculate a hash for the dependencies."""
        dependency_hashes = []

        def checkattr(name):
            def check(obj):
                try:
                    attrgetter(name)(obj)
                    return True
                except AttributeError:
                    return False

            return check

        for f in dependencies:
            code = None
            for flag_check, retrieve_func, retrieve_code in (
                # TODO: Make this more robust than simply relying on this ordering.
                # The ordering here is very important since functools.wraps() will
                # copy the '_dependency' flag.
                (
                    checkattr("_orig_func._dependency"),
                    attrgetter("_orig_func"),
                    attrgetter("_orig_func.__code__"),
                ),
                (checkattr("_dependency"), lambda f: f, attrgetter("__code__")),
            ):
                if not flag_check(f):
                    continue

                func = retrieve_func(f)
                code = retrieve_code(f)

                break
            else:
                raise ValueError("All dependencies must be marked with '_dependency'.")

            # Ensure that the hash can be calculated, i.e. that there are no mutable
            # objects present in the default arguments. Copy the object since some
            # object (e.g. immutabledict) will cache the hash resulting from calling
            # `hash()` (e.g. in a '_hash' attribute), and since the output of Python's
            # `hash()` function is not constant across sessions, this causes Joblib's
            # hash to change as well (which we do not want).
            hash(copy(signature(func)))
            # Finally, calculate the hash using Joblib because the inbuilt hash()
            # function changes its output in between runs.
            dependency_hashes.append(joblib.hashing.hash(signature(func)))
            dependency_hashes.append(joblib.hashing.hash(CodeObj(code).hashable()))

        dependency_hash = joblib.hashing.hash(dependency_hashes)
        return dependency_hash

    def __call__(self, *args, dependencies=(), **kwargs):
        dependency_hash = self.calculate_dependency_hash(dependencies)

        assert (
            len(args) == 1 and not kwargs
        ), "Only a function and dependencies should be given here."

        # The decorator was not configured with additional arguments.
        return self._decorator(args[0], dependency_hash=dependency_hash)

    def _get_hashed(self, func, *args, dependency_hash=None, **kwargs):
        assert dependency_hash is not None
        args, kwargs = extract_uniform_args_kwargs(func, *args, **kwargs)

        # Go through the original arguments and hash the contents manually.
        args_hashes = []
        for arg in args:
            args_hashes.append(self.hash_func(arg))

        # Repeat the above process for the kwargs. The keys should never include
        # MaskedArray data so we only need to deal with the values.
        kwargs_hashes = {}
        for key, arg in kwargs.items():
            kwargs_hashes[key] = self.hash_func(arg)

        # Include a hashed representation of the original function to ensure we can
        # tell different functions apart.
        func_code = CodeObj(func.__code__).hashable()

        return dict(
            func_code=func_code,
            args_hashes=args_hashes,
            kwargs_hashes=kwargs_hashes,
            dependencies=dependency_hash,
        )

    def _decorator(self, func, dependency_hash):
        def inner(hashed, args, kwargs):
            return func(*args, **kwargs)

        cached_inner = self.memory.cache(ignore=["args", "kwargs"])(inner)

        def bound_get_hashed(*orig_args, **orig_kwargs):
            return self._get_hashed(
                func, *orig_args, dependency_hash=dependency_hash, **orig_kwargs
            )

        @wraps(func)
        def cached_func(*orig_args, **orig_kwargs):
            hashed = bound_get_hashed(*orig_args, **orig_kwargs)
            return cached_inner(hashed, orig_args, orig_kwargs)

        return Decorated(
            cached_func=cached_func,
            cached_inner=cached_inner,
            bound_get_hashed=bound_get_hashed,
        )


_cache = DepMACache(memory=_memory, hash_func=custom_get_hash)


def mark_dependency(f):
    """Decorator which marks a function as a potential dependency.

    Args:
        f (callable): The dependency to be recorded.

    """
    f._dependency = True
    return f


def cache(*args, ma_cache_inst=_cache, dependencies=()):
    """A cached function with limited MaskedArray support.

    The keyword argument `cache_check` will be added and used automatically to
    determine whether a given call is already cached, if requested by the caller by
    giving `cache_check=True`. In this case, the cached function will return
    `IN_STORE` if the call is already cached, or raise a `NotCachedError` otherwise.

    Args:
        func (callable): Function to be cached.
        ma_cache_inst (wildfires.data.ma_cache): Cache instance defining the Joblib
            Memory instance to use for caching. Can be overriden e.g. for testing.
        dependencies (tuple of callable): Other functions the cached function depends
            on. If any of these functions change from one run to the next, the cache
            will be invalidated.
        is_dependency (bool): If True, this function will be marked as a possible
            dependency for other functions.

    """
    if not args:
        return partial(cache, ma_cache_inst=ma_cache_inst, dependencies=dependencies)

    assert callable(args[0])
    assert len(args) == 1

    func = args[0]

    # Update dependencies to enable chaining of dependencies.
    dependencies = (
        *dependencies,
        *reduce(
            add,
            (list(getattr(dep_func, "_dependencies", [])) for dep_func in dependencies),
            [],
        ),
    )

    cached_func = ma_cache_inst(func, dependencies=dependencies)

    @wraps(func)
    def cached_check(*args, cache_check=False, **kwargs):
        if cache_check:
            if not cached_func.is_cached(*args, **kwargs):
                raise NotCachedError("The given call is not cached.")
            return IN_STORE
        # Otherwise continue on as normal.
        return cached_func(*args, **kwargs)

    cached_check._orig_func = func
    cached_check._dependencies = dependencies

    return cached_check


def check_in_store(cached_func, *args, **kwargs):
    """Convenience function to check whether a given call is already cached."""
    if cached_func(*args, cache_check=True, **kwargs) is not IN_STORE:
        raise InvalidCacheCheck(
            f"The function {cached_func} is misconfigured for cache checking."
        )
    return IN_STORE
