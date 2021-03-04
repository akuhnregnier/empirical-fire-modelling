# -*- coding: utf-8 -*-
from functools import wraps

from wildfires.logging_config import LOGGING
from wildfires.logging_config import enable_logging as _enable_logging


@wraps(_enable_logging)
def enable_logging(*args, **kwargs):
    # Configure the empirical_fire_modelling logger by cloning the 'wildfires'
    # configuration. This will mean that all log files are consolidated under the
    # 'wildfires' name.
    LOGGING["loggers"]["empirical_fire_modelling"] = LOGGING["loggers"]["wildfires"]
    _enable_logging(*args, **kwargs)
