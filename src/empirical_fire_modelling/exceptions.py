# -*- coding: utf-8 -*-
class NotCachedError(ValueError):
    """Raised when the given call is not cached."""


class NoCX1Error(ValueError):
    """Raised when running on CX1 was requested but not allowed."""
