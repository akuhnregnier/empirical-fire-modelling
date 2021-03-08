# -*- coding: utf-8 -*-
"""LOCO calculation."""

from wildfires.dask_cx1 import dask_fit_loco

from ..cache import cache


@cache(ignore=["client", "local_n_jobs"])
def calculate_loco(
    rf, X_train, y_train, X_test, y_test, leave_out, client, local_n_jobs=31
):
    """Calculate the LOCO importances."""
    return dict(
        dask_fit_loco(
            estimator=rf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            client=client,
            leave_out=leave_out,
            local_n_jobs=local_n_jobs,
            verbose=True,
        )
    )
