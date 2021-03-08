# -*- coding: utf-8 -*-
"""VIF calculation."""

from wildfires.analysis import vif

from ..cache import cache


@cache
def calculate_vif(X):
    """Calculate the VIF."""
    return vif(X, verbose=True).set_index("Name", drop=True).T
