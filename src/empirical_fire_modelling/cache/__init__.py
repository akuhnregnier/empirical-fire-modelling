# -*- coding: utf-8 -*-
from functools import partial

from wildfires.cache import (
    IN_STORE,
    cache_hash_value,
    get_proxied_estimator,
    mark_dependency,
    process_proxy,
)

from .core import VariableProxyMemory, cache, memory

process_proxy = partial(process_proxy, memory=memory)
cache_hash_value = partial(cache_hash_value, hash_func=memory.get_hash)
get_proxied_estimator = partial(get_proxied_estimator, memory=memory)
