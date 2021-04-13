# -*- coding: utf-8 -*-
import logging

from joblib import register_store_backend
from wildfires.cache.proxy_backend import ProxyStoreBackend

from .hashing import OffsetVariableHasher, VariableHasher, VariableSequenceHasher

logger = logging.getLogger(__name__)


def register_backend():
    """Register variable proxy backend for use with joblib memory."""
    logger.debug("Registering variable-proxy Joblib backend.")
    register_store_backend("variable-proxy", VariableProxyBackend)


class VariableProxyBackend(ProxyStoreBackend):
    initial_hashers = ProxyStoreBackend.initial_hashers
    context_managers = ProxyStoreBackend.context_managers
    guarded_hashers = ProxyStoreBackend.guarded_hashers + [
        OffsetVariableHasher(),
        VariableHasher(),
        VariableSequenceHasher(),
    ]
