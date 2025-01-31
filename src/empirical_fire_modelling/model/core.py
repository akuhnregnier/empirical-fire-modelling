# -*- coding: utf-8 -*-
"""Model training."""
from functools import partial

import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import mean_squared_error, r2_score
from wildfires.dask_cx1 import DaskRandomForestRegressor
from wildfires.qstat import get_ncpus

from ..cache import cache, cache_hash_value, mark_dependency
from ..configuration import param_dict

__all__ = (
    "assign_n_jobs",
    "get_gini_importances",
    "get_model",
    "get_model_predict",
    "get_model_scores",
    "threading_get_model_predict",
)


@cache(ignore=["parallel_backend_call"])
def _get_model(X_train, y_train, param_dict=param_dict, parallel_backend_call=None):
    """Perform model fitting."""
    model = DaskRandomForestRegressor(**param_dict)
    if parallel_backend_call is None:
        with parallel_backend("dask"):
            model.fit(X_train, y_train)
    else:
        with parallel_backend_call():
            model.fit(X_train, y_train)
    return model


def assign_n_jobs(model):
    """Assign `n_jobs` to the number of currently available CPUs."""
    model.n_jobs = get_ncpus()
    return model


def get_model(X_train, y_train, cache_check=False, **kwargs):
    """Perform model fitting if needed and set the `n_jobs` parameter."""
    if cache_check:
        return _get_model.check_in_store(X_train, y_train)
    return cache_hash_value(_get_model(X_train, y_train, **kwargs), func=assign_n_jobs)


@cache(ignore=["parallel_backend_call"])
@mark_dependency
def get_model_predict(
    *, X_train, y_train, parallel_backend_call, predict_X, param_dict=param_dict
):
    """Cached model prediction."""
    model = DaskRandomForestRegressor(**param_dict)
    with parallel_backend_call():
        model.fit(X_train, y_train)
    return model.predict(predict_X)


def threading_get_model_predict(*, cache_check=False, **kwargs):
    """Cached model prediction with the local threading backend."""
    kwargs["parallel_backend_call"] = (
        # Use local threading backend.
        partial(parallel_backend, "threading", n_jobs=get_ncpus())
    )
    if cache_check:
        return get_model_predict.check_in_store(**kwargs)
    return get_model_predict(**kwargs)


@cache
def get_model_scores(model, X_test, X_train, y_test, y_train):
    # XXX: Get train OOB score (check Dask impl.), train CV score
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


@cache
def get_gini_importances(X_train, y_train, param_dict=param_dict, **kwargs):
    rf = get_model(X_train, y_train, **kwargs)
    ind_trees_gini = pd.DataFrame(
        [tree.feature_importances_ for tree in rf],
        columns=X_train.columns,
    )
    mean_importances = ind_trees_gini.mean().sort_values(ascending=False)
    std_importances = ind_trees_gini.std().reindex(mean_importances.index, axis=1)
    return mean_importances, std_importances
