# -*- coding: utf-8 -*-
"""Caching using the custom backend."""
from wildfires.cache import ProxyMemory, get_memory
from wildfires.cache.proxy_backend import Factory, HashProxy

from .variable_proxy_backend import register_backend as register_variable_proxy_backend


class VariableProxyMemory(ProxyMemory):
    def __init__(self, location, **kwargs):
        register_variable_proxy_backend()
        self.memory = get_memory(location, backend="variable-proxy", **kwargs)


memory = VariableProxyMemory("empirical_fire_modelling", verbose=0)
cache = memory.cache


class ShapeHashProxy(HashProxy):
    """Lazy array-like proxy containing a pre-calculated hash value."""

    __slots__ = "shape"

    def __init__(self, *args, cached_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = cached_shape


@cache
def get_shape(X):
    return X.shape


def add_cached_shape(X):
    """Cache `X.shape`."""

    def get_data():
        return X

    return ShapeHashProxy(
        Factory(get_data),
        memory.get_hash,
        hash_value=memory.get_hash(X),
        cached_shape=get_shape(X),
    )
