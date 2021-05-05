# -*- coding: utf-8 -*-
"""Model fitting."""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
from loguru import logger as loguru_logger

from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.cx1 import run
from empirical_fire_modelling.data import (
    generate_structure,
    get_data,
    get_endog_exog_mask,
    random_binary_dilation_split,
)
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model, get_model_scores
from empirical_fire_modelling.utils import optional_client_call

mpl.rc_file(Path(__file__).resolve().parent.parent / "matplotlibrc")

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging(level="WARNING")

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)


structures = (
    generate_structure(3, 1),
    # generate_structure(3, 2),
    generate_structure(5, 2),
    # generate_structure(5, 3),
    generate_structure(5, 4),
    # generate_structure(7, 3),
    generate_structure(7, 4),
    generate_structure(7, 5),
    # generate_structure(7, 6),
)


def fit_random_binary_dilation(
    experiment, structure, test_frac, seed, cache_check=False, **kwargs
):
    if cache_check:
        get_data(experiment, cache_check=True)

    endog_data, exog_data, master_mask = get_endog_exog_mask(experiment)

    split_kwargs = dict(
        exog_data=exog_data,
        endog_data=endog_data,
        master_mask=master_mask,
        structure=structure,
        test_frac=test_frac,
        seed=seed,
        verbose=False,
    )
    if cache_check:
        random_binary_dilation_split.check_in_store(**split_kwargs)
    (
        desc_str,
        data_info,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = random_binary_dilation_split(**split_kwargs)

    model = optional_client_call(
        get_model,
        dict(X_train=X_train, y_train=y_train),
        cache_check=cache_check,
    )[0]

    if cache_check:
        return get_model_scores.check_in_store(model, X_test, X_train, y_test, y_train)
    return data_info, get_model_scores(model, X_test, X_train, y_test, y_train)


if __name__ == "__main__":
    cx1_kwargs = dict(walltime="04:00:00", ncpus=32, mem="60GB")
    experiments = list(Experiment)
    args = []
    for experiment in experiments:
        for structure_info, structure in structures:
            for test_frac in [0.1, 0.05, 0.01]:
                for seed in range(4):
                    args.append((experiment, structure, test_frac, seed))
    output = run(fit_random_binary_dilation, *zip(*args), cx1_kwargs=cx1_kwargs)

    from pprint import pprint

    pprint(output)
