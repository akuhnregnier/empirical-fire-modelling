# -*- coding: utf-8 -*-
from joblib import parallel_backend
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from wildfires.dask_cx1 import DaskRandomForestRegressor
from wildfires.qstat import get_ncpus

from ..cache import cache
from ..configuration import n_splits, param_dict


@cache
def fit_combination(X, y, combination, split_index):
    train_indices, test_indices = zip(
        *KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X)
    )

    X = X[list(combination)].to_numpy()
    y = y.to_numpy()

    assert X.shape[1] == 15

    X_train = X[train_indices[split_index]]
    y_train = y[train_indices[split_index]]

    X_test = X[test_indices[split_index]]
    y_test = y[test_indices[split_index]]

    scores = {}

    with parallel_backend("threading", n_jobs=get_ncpus()):
        rf = DaskRandomForestRegressor(**param_dict)
        rf.fit(X_train, y_train)

        y_test_pred = rf.predict(X_test)
        scores[("test_score", split_index)] = {
            "r2": r2_score(y_true=y_test, y_pred=y_test_pred),
            "mse": mean_squared_error(y_true=y_test, y_pred=y_test_pred),
        }

        y_train_pred = rf.predict(X_train)
        scores[("train_score", split_index)] = {
            "r2": r2_score(y_true=y_train, y_pred=y_train_pred),
            "mse": mean_squared_error(y_true=y_train, y_pred=y_train_pred),
        }

    return scores
