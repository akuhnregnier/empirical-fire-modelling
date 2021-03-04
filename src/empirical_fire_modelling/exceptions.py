# -*- coding: utf-8 -*-
class NotCachedError(ValueError):
    """Raised when the given call is not cached."""


class InvalidCacheCheck(RuntimeError):
    """Raised when a function is misconfigured for cache checking."""
