# -*- coding: utf-8 -*-
"""LOCO calculation."""

from wildfires.dask_cx1 import dask_fit_loco

from ..cache import cache


@cache
def calculate_loco(
    rf, X_train, y_train, X_test, y_test, client, leave_out, local_n_jobs=31
):
    """Calculate the LOCO importances."""
    return dict(
        dask_fit_loco(
            rf,
            X_train,
            y_train,
            client,
            leave_out,
            local_n_jobs=local_n_jobs,
            verbose=True,
        )
    )
