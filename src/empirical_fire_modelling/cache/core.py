# -*- coding: utf-8 -*-
"""Caching using the custom backend."""
from wildfires.cache import ProxyMemory, get_memory

from .variable_proxy_backend import register_backend as register_variable_proxy_backend


class VariableProxyMemory(ProxyMemory):
    def __init__(self, location, **kwargs):
        register_variable_proxy_backend()
        self.memory = get_memory(location, backend="variable-proxy", **kwargs)


memory = VariableProxyMemory("empirical_fire_modelling", verbose=0)
cache = memory.cache
