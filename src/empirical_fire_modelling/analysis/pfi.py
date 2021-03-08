# -*- coding: utf-8 -*-
"""PFI calculation."""

import eli5
from wildfires.qstat import get_ncpus

from ..cache import cache


@cache
def calculate_pfi(rf, X, y):
    """Calculate the PFI."""
    rf.n_jobs = get_ncpus()
    perm_importance = eli5.sklearn.PermutationImportance(rf, random_state=1).fit(X, y)
    return eli5.explain_weights_df(perm_importance, feature_names=list(X.columns))
