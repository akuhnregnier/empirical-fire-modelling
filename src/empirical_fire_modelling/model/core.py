# -*- coding: utf-8 -*-
"""Model training."""

from joblib import parallel_backend
from sklearn.metrics import mean_squared_error, r2_score
from wildfires.dask_cx1 import DaskRandomForestRegressor
from wildfires.qstat import get_ncpus

from ..cache import cache, check_in_store

__all__ = (
    "get_model",
    "get_model_scores",
)


@cache
def _get_model(X_train, y_train, param_dict):
    """Perform model fitting."""
    model = DaskRandomForestRegressor(**param_dict)
    with parallel_backend("dask"):
        model.fit(X_train, y_train)
    return model


def get_model(X_train, y_train, param_dict, cache_check=False):
    """Perform model fitting if needed and set the `n_jobs` parameter."""
    if cache_check:
        return check_in_store(_get_model, X_train, y_train, param_dict)
    model = _get_model(X_train, y_train, param_dict)
    model.n_jobs = get_ncpus()
    return model


@cache
def get_model_scores(model, X_test, X_train, y_test, y_train):
    model.n_jobs = get_ncpus()

    with parallel_backend("threading", n_jobs=get_ncpus()):
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

    return {
        "test_r2": r2_score(y_test, y_pred),
        "test_mse": mean_squared_error(y_test, y_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "oob_r2": model.oob_score_,
    }
