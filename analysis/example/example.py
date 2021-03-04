# -*- coding: utf-8 -*-
"""Typical initialisation and running of imported, cached functions.

Note the cached function cannot be defined here (i.e. in __main__) because this would
interfere with Joblib's automatic caching mechanisms (if applicable).

"""
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from example_func import cached_example_function
from loguru import logger as loguru_logger

from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.logging_config import enable_logging

if "TQDMAUTO" in os.environ:
    pass
else:
    pass

mpl.rc_file(Path(__file__).resolve().parent / "matplotlibrc")

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)


def calling_cached(x):
    return cached_example_function(x)


if __name__ == "__main__":
    # Relevant if called with the command 'cx1' instead of 'local'.
    cx1_kwargs = dict(walltime="01:00:00", ncpus=1, mem="1GB")

    # This works both with single jobs...
    run(calling_cached, (1,), cx1_kwargs=cx1_kwargs)

    # ... and array jobs.
    run(calling_cached, (2, 3, 4), cx1_kwargs=cx1_kwargs)
