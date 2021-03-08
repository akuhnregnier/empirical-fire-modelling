# -*- coding: utf-8 -*-
"""SHAP value analysis."""
import math

import numpy as np
import shap
from wildfires.utils import match_shape

from ..cache import cache
from ..configuration import shap_job_samples
from ..utils import get_mm_indices, tqdm


def get_shap_params(X_train):
    raw_max_index = X_train.shape[0] / shap_job_samples
    max_index = math.floor(raw_max_index)
    if abs(raw_max_index - max_index) < 1e-5:
        # The two are identical, meaning that the last slice of data would be empty.
        max_index -= 1

    shap_params = dict(
        job_samples=shap_job_samples,
        max_index=max_index,
    )
    # Upper bound.
    shap_params["total_samples"] = (shap_params["max_index"] + 1) * shap_params[
        "job_samples"
    ]
    return shap_params


@cache
def get_shap_values(rf, X, data=None, interaction=False):
    """Calculate SHAP values for `X`.
    When `data` is None, `feature_perturbation='tree_path_dependent'` by default.
    """
    if data is None:
        feature_perturbation = "tree_path_dependent"
    else:
        feature_perturbation = "interventional"

    explainer = shap.TreeExplainer(
        rf, data=data, feature_perturbation=feature_perturbation
    )

    if interaction:
        return explainer.shap_interaction_values(X)
    return explainer.shap_values(X)


@cache
def calculate_2d_masked_shap_values(
    X_train,
    master_mask,
    shap_values,
    kind="train",
    additional_mask=None,
):
    shap_results = {}

    def time_abs_max(x):
        out = np.take_along_axis(
            x, np.argmax(np.abs(x), axis=0).reshape(1, *x.shape[1:]), axis=0
        )
        assert out.shape[0] == 1
        return out[0]

    agg_keys, agg_funcs = zip(
        ("masked_shap_arrs", lambda arr: np.mean(arr, axis=0)),
        ("masked_shap_arrs_std", lambda arr: np.std(arr, axis=0)),
        ("masked_abs_shap_arrs", lambda arr: np.mean(np.abs(arr), axis=0)),
        ("masked_abs_shap_arrs_std", lambda arr: np.std(np.abs(arr), axis=0)),
        ("masked_max_shap_arrs", time_abs_max),
    )
    for key in agg_keys:
        shap_results[key] = dict(data=[], vmins=[], vmaxs=[])

    mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices = get_mm_indices(
        master_mask
    )
    if kind == "train":
        mm_kind_indices = mm_valid_train_indices[: shap_values.shape[0]]
    elif kind == "val":
        mm_kind_indices = mm_valid_val_indices[: shap_values.shape[0]]
    else:
        raise ValueError(f"Unknown kind: {kind}.")

    for i in tqdm(range(len(X_train.columns)), desc="Aggregating SHAP values"):
        # Convert 1D shap values into 3D array (time, lat, lon).
        masked_shap_comp = np.ma.MaskedArray(
            np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
        )
        masked_shap_comp.ravel()[mm_kind_indices] = shap_values[:, i]

        if additional_mask is not None:
            masked_shap_comp.mask |= match_shape(
                additional_mask, masked_shap_comp.shape
            )

        # Calculate different aggregations over time.

        for key, agg_func in zip(agg_keys, agg_funcs):
            agg_shap = agg_func(masked_shap_comp)
            shap_results[key]["data"].append(agg_shap)
            shap_results[key]["vmins"].append(np.min(agg_shap))
            shap_results[key]["vmaxs"].append(np.max(agg_shap))

    # Calculate relative standard deviations.

    rel_agg_keys = [
        "masked_shap_arrs_rel_std",
        "masked_abs_shap_arrs_rel_std",
    ]
    rel_agg_sources_keys = [
        ("masked_shap_arrs", "masked_shap_arrs_std"),
        ("masked_abs_shap_arrs", "masked_abs_shap_arrs_std"),
    ]
    for rel_agg_key, rel_agg_sources_key in zip(rel_agg_keys, rel_agg_sources_keys):
        shap_results[rel_agg_key] = dict(data=[], vmins=[], vmaxs=[])
        for i in range(len(X_train.columns)):
            rel_agg_shap = shap_results[rel_agg_sources_key[1]]["data"][i] / np.ma.abs(
                shap_results[rel_agg_sources_key[0]]["data"][i]
            )
            shap_results[rel_agg_key]["data"].append(rel_agg_shap)
            shap_results[rel_agg_key]["vmins"].append(np.min(rel_agg_shap))
            shap_results[rel_agg_key]["vmaxs"].append(np.max(rel_agg_shap))

    for key, values in shap_results.items():
        values["vmin"] = min(values["vmins"])
        values["vmax"] = max(values["vmaxs"])

    return shap_results
