# -*- coding: utf-8 -*-
"""SHAP value analysis."""
import math
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from wildfires.utils import match_shape, shorten_features, significant_peak

from ..cache import cache
from ..configuration import shap_job_samples
from ..plotting import cube_plotting, get_sci_format
from ..utils import get_mm_indices, tqdm


def get_shap_params(X):
    raw_max_index = X.shape[0] / shap_job_samples
    # This is inclusive (PBS).
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


@cache(ignore=["verbose"])
def get_shap_values(rf, X, data=None, interaction=False, verbose=True):
    """Calculate SHAP values for `X`.

    When `data` is None, `feature_perturbation='tree_path_dependent'` by default.

    If `X.shape[0] > configuration.shap_job_samples`, this function will chunk `X`
    along its first axis und call `get_shap_values` using these smaller chunks
    repeatedly until SHAP values for all chunks have been computed. It will then
    return the collated results.

    """
    if data is not None:
        raise NotImplementedError(
            "Would have to implement more, e.g. chunking of `data`."
        )

    if data is None:
        feature_perturbation = "tree_path_dependent"
    else:
        feature_perturbation = "interventional"

    shap_params = get_shap_params(X)
    max_chunk_index = shap_params["max_index"] + 1  # Add 1 (non-inclusive upper lim).
    if max_chunk_index > 1:
        shap_arrs = []
        for start_index in tqdm(
            range(max_chunk_index), desc="SHAP value chunks", disable=not verbose
        ):
            shap_arrs.append(
                get_shap_values(
                    rf,
                    X.iloc[
                        start_index
                        * shap_params["job_samples"] : (start_index + 1)
                        * shap_params["job_samples"]
                    ],
                    data=data,
                    interaction=interaction,
                )
            )
        return np.vstack(shap_arrs)

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
    kind="val",
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


def plot_shap_value_maps(
    X_train, shap_results, map_figure_saver, directory="shap_maps", close=True
):
    """

    Args:
        X_train (pandas DataFrame):
        shap_results (SHAP results dict from `calculate_2d_masked_shap_values`):
        map_figure_saver (FigureSaver instance):
        directory (str or Path): Figure saving directory.
        close (bool): If True, close figures after saving.

    """
    # Define common plotting profiles, as `cube_plotting` kwargs.

    def get_plot_kwargs(feature, results_dict, title_stub, kind=None):
        kwargs = dict(
            fig=plt.figure(figsize=(5.1, 2.8)),
            title=f"{title_stub} '{shorten_features(feature)}'",
            nbins=7,
            vmin=results_dict["vmin"],
            vmax=results_dict["vmax"],
            log=True,
            log_auto_bins=False,
            extend="neither",
            min_edge=1e-3,
            cmap="inferno",
            colorbar_kwargs=dict(
                format=get_sci_format(ndigits=1, atol_exceeded="adjust"),
                label=f"SHAP ('{shorten_features(str(feature))}')",
            ),
            coastline_kwargs={"linewidth": 0.3},
        )
        if kind == "mean":
            kwargs.update(
                cmap="Spectral_r",
                cmap_midpoint=0,
                cmap_symmetric=True,
            )
        if kind == "rel_std":
            kwargs.update(
                vmin=1e-2,
                vmax=10,
                extend="both",
                nbins=5,
            )
        return kwargs

    for i, feature in enumerate(tqdm(X_train.columns, desc="Mapping SHAP values")):
        for agg_key, title_stub, kind, sub_directory in (
            ("masked_shap_arrs", "Mean SHAP value for", "mean", "mean"),
            ("masked_shap_arrs_std", "STD SHAP value for", None, "std"),
            (
                "masked_shap_arrs_rel_std",
                "Rel STD SHAP value for",
                "rel_std",
                "rel_std",
            ),
            ("masked_abs_shap_arrs", "Mean |SHAP| value for", None, "abs_mean"),
            ("masked_abs_shap_arrs_std", "STD |SHAP| value for", None, "abs_std"),
            (
                "masked_abs_shap_arrs_rel_std",
                "Rel STD |SHAP| value for",
                "rel_std",
                "rel_abs_std",
            ),
            ("masked_max_shap_arrs", "Max || SHAP value for", "mean", "max"),
        ):
            fig = cube_plotting(
                shap_results[agg_key]["data"][i],
                **get_plot_kwargs(
                    feature,
                    results_dict=shap_results[agg_key],
                    title_stub=title_stub,
                    kind=kind,
                ),
            )
            map_figure_saver.save_figure(
                fig,
                f"{agg_key}_{feature}",
                sub_directory=Path(directory) / sub_directory,
            )
            if close:
                plt.close(fig)


@cache
def get_max_positions(
    *,
    X,
    variables,
    shap_results,
    shap_measure,
    mean_ba,
    exclude_inst,
    ptp_threshold_factor,
    diff_threshold,
):
    """Using SHAP values, get weighted average of lags."""
    lags = tuple(v.shift for v in variables)
    # Ensure lags are sorted consistently.
    assert list(lags) == sorted(lags)

    if exclude_inst and 0 in lags:
        assert lags[0] == 0
        lags = lags[1:]
        variables = variables[1:]

    n_features = len(variables)

    # There is no point plotting this map for a single feature or less since we are
    # interested in a comparison between different feature ranks.
    if n_features <= 1:
        raise ValueError(f"Too few features: {n_features}.")

    selected_data = np.empty(n_features, dtype=object)
    for i, var in enumerate(X.columns):
        if var in variables:
            selected_data[lags.index(var.shift)] = shap_results[shap_measure]["data"][
                i
            ].copy()

    shared_mask = reduce(np.logical_or, (data.mask for data in selected_data))
    for data in selected_data:
        data.mask = shared_mask

    stacked_shaps = np.vstack([data.data[np.newaxis] for data in selected_data])

    # Calculate the significance of the global maxima for each of the valid pixels.

    # Valid indices are recorded in 'shared_mask'.

    valid_i, valid_j = np.where(~shared_mask)

    max_positions = np.ma.MaskedArray(
        np.zeros_like(shared_mask, dtype=np.float64), mask=True
    )
    for i, j in zip(tqdm(valid_i, desc="Evaluating maxima", smoothing=0), valid_j):
        ptp_threshold = ptp_threshold_factor * mean_ba[i, j]
        if significant_peak(
            stacked_shaps[:, i, j],
            diff_threshold=diff_threshold,
            ptp_threshold=ptp_threshold,
        ):
            # If the maximum is significant, go on the calculate the weighted avg. of the signal.
            max_positions[i, j] = np.average(
                lags, weights=np.abs(stacked_shaps[:, i, j])
            )
    return max_positions
