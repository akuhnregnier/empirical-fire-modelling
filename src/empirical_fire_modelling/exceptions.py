# -*- coding: utf-8 -*-
class NoCX1Error(ValueError):
    """Raised when running on CX1 was requested but not allowed."""


class EmptyUnitSpecError(ValueError):
    """Raised when an empty units spec is found in a label."""
