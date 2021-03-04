# -*- coding: utf-8 -*-
"""Example function."""
from empirical_fire_modelling.cache import cache


@cache
def cached_example_function(x):
    print("Got:", x)
    return x + 1
