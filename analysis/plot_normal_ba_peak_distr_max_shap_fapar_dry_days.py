# -*- coding: utf-8 -*-
"""15VEG_FAPAR SHAP peak distribution plotting."""
import logging
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import combinations, islice
from pathlib import Path
from string import ascii_lowercase

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iris.time import PartialDateTime
from loguru import logger as loguru_logger
from matplotlib.colors import LinearSegmentedColormap, from_levels_and_colors
from scipy.ndimage import binary_dilation
from wildfires.data import Ext_ESA_CCI_Landcover_PFT
from wildfires.utils import (
    get_centres,
    get_masked_array,
    get_unmasked,
    shorten_features,
    significant_peak,
)

from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.shap import (
    calculate_2d_masked_shap_values,
    get_shap_values,
    sort_peaks,
)
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.data import get_endog_exog_mask, get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.plotting import cube_plotting  # XXX temporary
from empirical_fire_modelling.plotting import map_figure_saver
from empirical_fire_modelling.utils import check_master_masks, tqdm

mpl.rc_file(Path(__file__).resolve().parent / "matplotlibrc")

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


if __name__ == "__main__":
    experiment = Experiment["15VEG_FAPAR"]

    # Operate on cached model / data only.
    get_endog_exog_mask.check_in_store(experiment)
    endog_data, _, master_mask = get_endog_exog_mask(experiment)

    check_master_masks(master_mask)

    get_experiment_split_data.check_in_store(experiment)
    X_train, X_test, y_train, y_test = get_experiment_split_data(experiment)

    get_model(X_train, y_train, cache_check=True)
    rf = get_model(X_train, y_train)

    get_shap_values.check_in_store(rf=rf, X=X_test)
    shap_values = get_shap_values(rf=rf, X=X_test)

    # Analysis / plotting parameters.
    diff_threshold = 0.5
    ptp_threshold_factor = 0.12  # relative to the mean

    chosen_lags = tuple(lag for lag in variable.lags if lag <= 9)
    assert list(chosen_lags) == sorted(chosen_lags)

    map_shap_results = calculate_2d_masked_shap_values(
        X_train, master_mask, shap_values, kind="val"
    )

    target_ba = get_masked_array(endog_data.values, master_mask)
    mean_ba = np.ma.mean(target_ba, axis=0)

    # def param_iter():
    #     for variable_factory in tqdm(
    #         [variable.FAPAR, variable.DRY_DAY_PERIOD], desc="Feature"
    #     ):
    #         for exclude_inst in tqdm([False, True], desc="Exclude inst."):
    #             yield exclude_inst, variable_factory

    # weighted_plot_data = {}
    # for exclude_inst, variable_factory in param_iter():
    #     weighted_plot_data[(exclude_inst, variable_factory)] = get_max_positions(
    #         X=X_test,
    #         variables=[variable_factory[lag] for lag in chosen_lags],
    #         shap_results=map_shap_results,
    #         shap_measure="masked_max_shap_arrs",
    #         mean_ba=mean_ba,
    #         exclude_inst=exclude_inst,
    #         ptp_threshold_factor=ptp_threshold_factor,
    #         diff_threshold=diff_threshold,
    #     )

    # XXX Calculation start.
    pfts = Ext_ESA_CCI_Landcover_PFT()
    pfts.limit_months(start=PartialDateTime(2010, 1), end=PartialDateTime(2015, 1))
    pfts.regrid()
    pfts = pfts.get_mean_dataset()

    max_month = 9
    close_figs = True
    verbose = 2

    filter_name = "normal"
    shap_results = map_shap_results
    shap_measure = "masked_max_shap_arrs"

    X = X_test

    def param_iter():
        for variable_factory in tqdm(
            [variable.FAPAR, variable.DRY_DAY_PERIOD],
            desc="Feature",
            disable=verbose < 1,
        ):
            for (exc_name, exclude_inst) in tqdm(
                [("with_inst", False), ("no_inst", True)],
                desc="Exclude inst.",
                disable=verbose < 2,
            ):
                yield (exc_name, exclude_inst), variable_factory

    peak_data_dict = {}
    for ((exc_name, exclude_inst), variable_factory) in param_iter():
        filtered_vars = [variable_factory[i] for i in chosen_lags]
        filtered_lags = chosen_lags

        if exclude_inst and 0 in filtered_lags:
            assert filtered_lags[0] == 0
            filtered_lags = filtered_lags[1:]
            filtered_vars = filtered_vars[1:]

        n_features = len(filtered_vars)

        # There is no point plotting this map for a single feature or less since we are
        # interested in a comparison between different feature ranks.
        if n_features <= 1:
            raise ValueError("Not enough features.")

        selected_data = np.empty(n_features, dtype=object)
        for i, var in enumerate(X_train.columns):
            if var in filtered_vars:
                selected_data[filtered_lags.index(var.shift)] = shap_results[
                    shap_measure
                ]["data"][i].copy()

        shared_mask = reduce(np.logical_or, (data.mask for data in selected_data))
        for data in selected_data:
            data.mask = shared_mask

        stacked_shaps = np.vstack([data.data[np.newaxis] for data in selected_data])

        # Calculate the significance of the global maxima for each of the valid pixels.

        # Valid indices are recorded in 'shared_mask'.

        valid_i, valid_j = np.where(~shared_mask)
        total_valid = len(valid_i)

        peak_indices = []

        for i, j in zip(
            tqdm(valid_i, desc="Evaluating maxima", smoothing=0, disable=verbose < 3),
            valid_j,
        ):
            ptp_threshold = ptp_threshold_factor * mean_ba[i, j]
            peaks_i = significant_peak(
                stacked_shaps[:, i, j],
                diff_threshold=diff_threshold,
                ptp_threshold=ptp_threshold,
                strict=False,
            )

            # Adding information about the sign of the mean influence, sorted by time.
            peak_indices.append(
                tuple(
                    f"{filtered_lags[p_i]}({'+' if stacked_shaps[p_i, i, j] > 0 else '-'})"
                    for p_i in sorted(peaks_i)
                )
            )

        peak_data_dict[(exclude_inst, variable_factory)] = dict(
            filtered_vars=filtered_vars,
            filtered_lags=filtered_lags,
            n_features=n_features,
            valid_i=valid_i,
            valid_j=valid_j,
            total_valid=total_valid,
            peak_indices=peak_indices,
            shared_mask=shared_mask,
        )

    peak_data_dicts = []
    for ((exc_name, exclude_inst), variable_factory) in param_iter():
        short_feature = shorten_features(str(variable_factory))

        sub_directory = (
            Path("shap_peaks") / filter_name / shap_measure / short_feature / exc_name
        )

        valid_i = peak_data_dict[(exclude_inst, variable_factory)]["valid_i"]
        valid_j = peak_data_dict[(exclude_inst, variable_factory)]["valid_j"]
        peak_indices = peak_data_dict[(exclude_inst, variable_factory)]["peak_indices"]
        shared_mask = peak_data_dict[(exclude_inst, variable_factory)]["shared_mask"]

        # Determine the number of peaks at each location.
        peaks_arr = np.ma.MaskedArray(
            np.zeros_like(shared_mask, dtype=np.float64), mask=True
        )
        for i, j, indices in zip(valid_i, valid_j, peak_indices):
            peaks_arr[i, j] = len(indices)

        # Exclude locations with too many peaks.
        masked_peaks = peaks_arr.copy()
        masked_peaks.mask |= (peaks_arr.data == 0) | (peaks_arr.data > 2)

        # Determine the peaks present at each valid location.
        valid_peak_indices = []
        for i, j, peaks_i in zip(valid_i, valid_j, peak_indices):
            if masked_peaks.mask[i, j]:
                # Only use valid samples.
                continue
            valid_peak_indices.append(peaks_i)

        assert np.all(
            np.sort(np.unique([len(indices) for indices in valid_peak_indices]))
            == np.array([1, 2])
        )

        peaks_dict = dict(zip(*np.unique(valid_peak_indices, return_counts=True)))

        total_counts = np.sum(list(peaks_dict.values()))
        relative_counts_dict = {
            key: val / total_counts for key, val in peaks_dict.items()
        }

        #
        # Limit the number of peak combinations.
        #

        keys, values = list(zip(*relative_counts_dict.items()))
        keys = np.asarray(keys)
        values = np.asarray(values)

        max_n_peaks = 6
        min_frac = 0.075  # Require at least this fraction per entry.

        sorted_indices = np.argsort(values)
        cumulative_fractions = np.cumsum(values[sorted_indices])

        # Ensure at least `min_n_entries` entries are present, but no more than `max_n_peaks`.
        mask = np.ones_like(cumulative_fractions, dtype=np.bool_)
        mask[:-max_n_peaks] = False  # no more than `max_n_peaks`.

        mask &= values[sorted_indices] > min_frac

        print(
            f"Remaining fraction: {short_feature}, exclude inst: {exclude_inst}",
            np.sum(values[sorted_indices][mask]),
        )

        thres_counts_dict = {
            key: val
            for key, val in zip(
                keys[sorted_indices][mask], values[sorted_indices][mask]
            )
        }
        print(f"{short_feature}, exclude inst: {exclude_inst}", thres_counts_dict)

        peak_keys = list(thres_counts_dict)

        imp_peaks = np.ma.MaskedArray(
            np.zeros_like(shared_mask, dtype=np.int64),
            mask=True,
        )
        for i, j, indices in zip(valid_i, valid_j, peak_indices):
            if indices in peak_keys:
                imp_peaks[i, j] = peak_keys.index(indices)

        assert len(np.unique(imp_peaks.data[~imp_peaks.mask])) == len(peak_keys)

        peak_results = {}
        for comb_i in tqdm(np.unique(get_unmasked(imp_peaks)), desc="Peak combination"):
            for pft_cube in pfts:
                selection = (~(pft_cube.data.mask | imp_peaks.mask)) & np.isclose(
                    imp_peaks, comb_i
                )
                peak_results[
                    (str(peak_keys[int(comb_i)]), pft_cube.name())
                ] = pft_cube.data.data[selection]

        peak_comb_df = pd.DataFrame(
            {key: pd.Series(vals) for key, vals in peak_results.items()}
        )
        peak_comb_df.columns.names = ["peak_combination", "pft"]

        peak_data_dicts.append(
            dict(
                exc_name=exc_name,
                exclude_inst=exclude_inst,
                short_feature=short_feature,
                sub_directory=sub_directory,
                filtered_vars=peak_data_dict[(exclude_inst, variable_factory)][
                    "filtered_vars"
                ],
                filtered_lags=peak_data_dict[(exclude_inst, variable_factory)][
                    "filtered_lags"
                ],
                n_features=peak_data_dict[(exclude_inst, variable_factory)][
                    "n_features"
                ],
                valid_i=valid_i,
                valid_j=valid_j,
                total_valid=peak_data_dict[(exclude_inst, variable_factory)][
                    "total_valid"
                ],
                peak_indices=peak_indices,
                peaks_arr=peaks_arr,
                masked_peaks=masked_peaks,
                valid_peak_indices=valid_peak_indices,
                peaks_dict=peaks_dict,
                total_counts=total_counts,
                relative_counts_dict=relative_counts_dict,
                thres_counts_dict=thres_counts_dict,
                peak_keys=peak_keys,
                imp_peaks=imp_peaks,
                peak_comb_df=peak_comb_df,
            )
        )

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]

        pd.Series(
            dict(
                zip(
                    *np.unique(
                        [len(indices) for indices in plot_data["peak_indices"]],
                        return_counts=True,
                    )
                )
            )
        ).plot.bar(
            ax=plt.subplots(figsize=(6, 4))[1],
            title=f"{short_feature}, {exc_name}",
            rot=0,
        )
        # figure_saver.save_figure(
        #     plt.gcf(), "n_peaks_distr", sub_directory=sub_directory
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]

        fig, cbar = cube_plotting(
            plot_data["peaks_arr"],
            title=f"Nr. Peaks {short_feature}, {exc_name}",
            boundaries=np.arange(0, 4) - 0.5,
            fig=plt.figure(figsize=(5.1, 2.6)),
            coastline_kwargs={"linewidth": 0.3},
            colorbar_kwargs={"label": "nr. peaks", "format": "%0.1f"},
            return_cbar=True,
        )

        tick_pos = np.arange(4, dtype=np.float64)
        tick_pos[3] -= 0.5

        cbar.set_ticks(tick_pos)

        tick_labels = list(map(str, range(3))) + [">2"]
        cbar.set_ticklabels(tick_labels)

        # map_figure_saver.save_figure(
        #     fig, f"nr_shap_peaks_map_{short_feature}", sub_directory=sub_directory
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]

        cmap, norm = from_levels_and_colors(
            levels=np.arange(1, 4) - 0.5,
            colors=["C1", "C2"],
            extend="neither",
        )

        fig, cbar = cube_plotting(
            plot_data["masked_peaks"],
            title=f"Nr. Peaks {short_feature}, {exc_name}",
            # boundaries=np.arange(1, 4) - 0.5,
            fig=plt.figure(figsize=(5.1, 2.6)),
            coastline_kwargs={"linewidth": 0.3},
            colorbar_kwargs={"label": "nr. peaks", "format": "%0.1f"},
            return_cbar=True,
            cmap=cmap,
            norm=norm,
        )

        tick_pos = np.arange(3, dtype=np.float64)
        cbar.set_ticks(tick_pos)

        tick_labels = list(map(str, range(1, 3)))
        cbar.set_ticklabels(tick_labels)

        #     plt.gca().gridlines()

        # map_figure_saver.save_figure(
        #     fig,
        #     f"filtered_nr_shap_peaks_map_{short_feature}",
        #     sub_directory=sub_directory,
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]

        pd.Series(
            dict(
                zip(
                    *np.unique(
                        [len(indices) for indices in plot_data["valid_peak_indices"]],
                        return_counts=True,
                    )
                )
            )
        ).plot.bar(
            ax=plt.subplots(figsize=(6, 4))[1],
            title=f"{short_feature}, {exc_name}",
            rot=0,
        )
        # figure_saver.save_figure(
        #     plt.gcf(), "filtered_n_peaks_distr", sub_directory=sub_directory
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]
        relative_counts_dict = plot_data["relative_counts_dict"]

        fig = plt.figure(figsize=(7, 0.3 * len(relative_counts_dict) + 0.4))
        pd.Series(
            {", ".join(k): v for k, v in relative_counts_dict.items()}
        ).sort_values().plot.barh(
            fontsize=12,
            title=f"{short_feature}, {exc_name}",
        )
        plt.grid(alpha=0.4, linestyle="--")
        # figure_saver.save_figure(
        #     plt.gcf(), "peak_comb_distr", sub_directory=sub_directory
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    for plot_data in peak_data_dicts:
        short_feature = plot_data["short_feature"]
        exc_name = plot_data["exc_name"]
        sub_directory = plot_data["sub_directory"]
        relative_counts_dict = plot_data["relative_counts_dict"]
        peak_keys = plot_data["peak_keys"]
        imp_peaks = plot_data["imp_peaks"]

        boundaries = np.arange(len(peak_keys) + 1) - 0.5

        cmap, norm = from_levels_and_colors(
            levels=boundaries,
            colors=[plt.get_cmap("tab10")(i) for i in range(len(peak_keys))],
            extend="neither",
        )

        fig, cbar = cube_plotting(
            imp_peaks,
            title=f"Peak Distr. {short_feature}, {exc_name}",
            fig=plt.figure(figsize=(5.1, 2.6)),
            coastline_kwargs={"linewidth": 0.3},
            colorbar_kwargs={"label": "peak combination"},
            return_cbar=True,
            cmap=cmap,
            norm=norm,
        )

        tick_pos = np.arange(len(peak_keys), dtype=np.float64)
        cbar.set_ticks(tick_pos)

        cbar.set_ticklabels(peak_keys)

        # map_figure_saver.save_figure(
        #     fig, f"shap_peak_distr_map_{short_feature}", sub_directory=sub_directory
        # )
        # if close_figs:
        #     plt.close()
    # XXX - needed?

    # XXX - needed?
    plot_data = list(islice(peak_data_dicts, 2, 3))[0]
    short_feature = plot_data["short_feature"]
    assert short_feature == "DD"
    exc_name = plot_data["exc_name"]
    assert exc_name == "with_inst"
    sub_directory = plot_data["sub_directory"]
    relative_counts_dict = plot_data["relative_counts_dict"]
    peak_keys = plot_data["peak_keys"]
    assert peak_keys == [("0(+)",), ("0(-)",)]
    raw_sel = imp_peaks = plot_data["imp_peaks"].copy()
    assert np.all(np.unique(get_unmasked(imp_peaks)) == np.array([0, 1]))
    assert np.all(np.unique(get_unmasked(raw_sel)) == np.array([0, 1]))

    sel = np.ma.MaskedArray(np.zeros_like(imp_peaks.data, dtype=np.int32), mask=True)

    # Dilate the 0s and 1s separately.
    for val in range(2):
        dilated = binary_dilation(
            (raw_sel == val).data.copy(), iterations=4, mask=imp_peaks.mask.copy()
        )
        sel[dilated.astype("bool")] = val

    plt.figure(figsize=(9, 5))
    plt.pcolormesh(sel)
    _ = plt.title(f"({short_feature}) Selection Overlap Template")
    # XXX - needed?

    # XXX - needed?
    def gen_cat_dict():
        return {
            # Associated with 0(+).
            "dd_0": set(),
            # Associated with 0(-).
            "dd_1": set(),
            # Mixed (see `cat_thres` below).
            "dd_mix": set(),
        }

    categories = defaultdict(gen_cat_dict)
    cat_thres = 0.7

    cat_masks = {}

    for plot_data in islice(peak_data_dicts, 0, None):
        short_feature = plot_data["short_feature"]
        peak_keys = plot_data["peak_keys"]
        imp_peaks = plot_data["imp_peaks"]

        overlap_data = {}
        for i in np.unique(get_unmasked(imp_peaks)):
            joined_key = "|".join(peak_keys[int(i)])

            new_mask = imp_peaks == i
            mask_key = (short_feature, joined_key)
            if mask_key in cat_masks:
                # Only overwrite an existing key if the new overlap region is larger.
                # This is relevant if the same peak (combination) exists with and
                # without inst for the same feature.
                if np.sum(new_mask) > np.sum(cat_masks[mask_key]):
                    cat_masks[mask_key] = new_mask
            else:
                # Assign new key.
                cat_masks[mask_key] = new_mask

            overlap_data[joined_key] = pd.Series(get_unmasked(sel[cat_masks[mask_key]]))

        df = pd.DataFrame(overlap_data)
        counts = df.apply(pd.Series.value_counts)
        counts /= counts.sum(axis=0)

        for name, data in counts.iteritems():
            above_thres = data > cat_thres
            if np.any(above_thres):
                if np.isclose(data.index[np.where(above_thres)[0][0]], 0):
                    categories[short_feature]["dd_0"].add(name)
                else:
                    categories[short_feature]["dd_1"].add(name)
            else:
                categories[short_feature]["dd_mix"].add(name)

        counts.columns.name = short_feature
        print(f"counts (exc_name: {plot_data['exc_name']}):\n", counts)

    cat_df = pd.DataFrame(categories)
    for _, peak_series in cat_df.iteritems():
        for peaks_a, peaks_b in combinations(peak_series, 2):
            # Each peak should only appear in a single category.
            assert not peaks_a.intersection(peaks_b)
    cat_df = cat_df.applymap(tuple)

    print("cat_df")
    print(cat_df)

    def total_count_list_items(s):
        count = 0
        for item in s:
            count += len(item)
        return count

    print("cat_df total")
    print(cat_df.apply(total_count_list_items))

    def count_list_items(s):
        counts = []
        for item in s:
            counts.append(len(item))
        return counts

    print("cat_df total")
    print(cat_df.apply(count_list_items))

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    cs_list = {
        # H:[86, 266], C:[20,100], L:[40, 95]
        "5 cool": [
            "#7b5a19",
            "#a4c906",
            "#d9c398",
            "#008f91",
            # "#0349fc"
            "#0e2db5",
        ],
        # H : 28, C : 123, L : 65
        "1 warm": ["#FB780F"],
        # H:[270, 10], C:[20, 100], L:[40, 95]
        "1 intermediate": [
            # "#ca99a5",
            "#e32266",
            # "#565a91",
        ],
    }

    def generate_cmap(cs):
        if len(cs) == 1:
            # Workaround to use the below method for a single color.
            cs = cs * 2
        return LinearSegmentedColormap.from_list("test", cs, N=len(cs))

    # Visualise the colormaps.
    fig, axes = plt.subplots(len(cs_list), figsize=(8, 3))
    for (ax, (title, cs)) in zip(axes, cs_list.items()):
        ax.imshow(gradient, aspect="auto", cmap=generate_cmap(cs))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(h_pad=0.3)

    category_map = {
        "dd_0": "1 warm",
        "dd_1": "5 cool",
        "dd_mix": "1 intermediate",
    }

    color_s = pd.Series(category_map).apply(lambda x: cs_list[x])

    # Build mapping from peaks to colors.

    # Each category (`dd_0`, etc...) has to be processed for both features
    # simultaneously to coordinate the assignment of the same colours to the most
    # similar (geographically) arrangement of peak patterns.

    peak_color_data = defaultdict(dict)

    for cat, peaks in cat_df.iterrows():
        # Get the colors corresponding to the chosen category (like 'dd_0').
        colors = deepcopy(color_s[cat])
        # Sort the corresponding peaks for each feature.
        peaks = deepcopy(peaks).apply(sort_peaks)

        # Number of shared peaks = lowest number of peaks.
        n_peaks = tuple(map(len, peaks))
        shared_n = min(n_peaks)

        if shared_n == 0 or all(n == 1 for n in n_peaks):
            # There is nothing to do - simply assign the colors to the peaks.
            for feature, peak_list in peaks.iteritems():
                for peak, color in zip(peak_list, colors):
                    peak_color_data[feature][peak] = color
            continue

        # There are shared peaks (both features have > 0). Determine which
        # of the colors to assign to which peaks to maximise the overlap.

        # Compute the overlap matrix.
        features = peaks.index
        shape = tuple(map(len, (peaks[feature] for feature in features)))
        overlaps = np.zeros(shape, dtype=np.int64)

        peak_lists = deepcopy(dict(peaks.iteritems()))

        for i, peak_i in enumerate(peaks[features[0]]):
            mask_i = cat_masks[(features[0], peak_i)]
            for j, peak_j in enumerate(peaks[features[1]]):
                mask_j = cat_masks[(features[1], peak_j)]
                overlaps[i, j] = np.sum(mask_i & mask_j)

        # Iteratively use the largest (remaining) overlap to guide color assignment.

        # We can only do this as many times as the shortest nr. of peaks.
        for _ in range(min(shape)):
            # Determine the largest overlap.

            # These two peaks will have the same color.
            max_indices = np.unravel_index(np.argmax(overlaps), shape)
            new_color = colors.pop()
            for i, new_feature in enumerate(features):
                new_peak = peaks[new_feature][max_indices[i]]
                peak_lists[new_feature].remove(new_peak)
                peak_color_data[new_feature][new_peak] = new_color

                # Ensure these peaks are not used again by setting the corresponding row / column to -1.
                full_slice = [slice(None)] * 2
                full_slice[i] = max_indices[i]
                overlaps[tuple(full_slice)] = -1

        # Fill out any remaining colours.
        for _ in range(max(map(len, peak_lists.values()))):
            new_color = colors.pop()
            for new_feature in features:
                if peak_lists[new_feature]:
                    # Only add a new color if there are still peaks to add colors to.
                    new_peak = peak_lists[new_feature].pop()
                    peak_color_data[new_feature][new_peak] = new_color

    peak_color_data = dict(peak_color_data)
    peak_color_df = pd.DataFrame(peak_color_data)
    peak_color_df = peak_color_df.reindex(sort_peaks(peak_color_df.index))

    # Visualise combined colorbars.

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(peak_color_df.shape[1], figsize=(8, 1.8))
    for (ax, (feature, color_s)) in zip(axes, peak_color_df.iteritems()):
        cs = color_s.dropna()

        # Separate peak keys into the categories.
        grouped_sorted_peak_keys = []
        for category in ["dd_0", "dd_1", "dd_mix"]:
            for peak_key in cs.index:
                if peak_key in cat_df[feature][category]:
                    grouped_sorted_peak_keys.append(peak_key)

        cs = cs.reindex(grouped_sorted_peak_keys)

        ax.imshow(gradient, aspect="auto", cmap=generate_cmap(cs))
        ax.set_title(feature)
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks(get_centres(np.linspace(0, 256, len(cs) + 1)))
        ax.set_xticklabels(cs.index)

    fig.tight_layout(h_pad=0.3)

    # XXX - needed?

    # XXX Calculation end.

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 5.2),
        subplot_kw=dict(projection=ccrs.Robinson()),
        dpi=300,
    )

    for ax, plot_data, title in zip(
        axes.T.ravel(),
        peak_data_dicts,
        ascii_lowercase,
    ):
        short_feature = plot_data["short_feature"]
        peak_keys = plot_data["peak_keys"]
        imp_peaks = plot_data["imp_peaks"].astype("int")
        exclude_inst = plot_data["exclude_inst"]

        boundaries = np.arange(len(peak_keys) + 1) - 0.5

        ungrouped_sorted_peak_keys = list(
            map(
                lambda p: tuple(p.split("|")),
                sort_peaks(tuple(map(lambda ps: "|".join(ps), peak_keys))),
            )
        )

        # Separate peak keys into the categories.
        grouped_sorted_peak_keys = []
        for category in ["dd_0", "dd_1", "dd_mix"]:
            for peak_key in ungrouped_sorted_peak_keys:
                if "|".join(peak_key) in cat_df[short_feature][category]:
                    grouped_sorted_peak_keys.append(peak_key)
        # Reverse to put the 'first' item on top of the colorbar.
        grouped_sorted_peak_keys = grouped_sorted_peak_keys[::-1]
        assert len(grouped_sorted_peak_keys) == len(ungrouped_sorted_peak_keys)

        # Map from the old peak indices to the new sorted indices.
        old_to_new = {}
        for old in np.unique(get_unmasked(imp_peaks)):
            old_to_new[old] = grouped_sorted_peak_keys.index(peak_keys[old])

        sorted_imp_peaks = imp_peaks.copy()
        for old, new in old_to_new.items():
            sorted_imp_peaks[imp_peaks == old] = new

        cmap, norm = from_levels_and_colors(
            levels=boundaries,
            colors=[
                peak_color_df[short_feature]["|".join(peaks)]
                for peaks in grouped_sorted_peak_keys
            ],
            extend="neither",
        )

        # XXX temporary
        cube_plotting(
            sorted_imp_peaks,
            title="",
            coastline_kwargs={"linewidth": 0.3},
            colorbar_kwargs=False,
            cmap=cmap,
            norm=norm,
            ax=ax,
        )

        if exclude_inst:
            exc_string = f"(no current {short_feature})"
        else:
            exc_string = f"(with current {short_feature})"
        ax.text(
            0.5,
            1.03,
            f"({title}) {short_feature} {exc_string}",
            transform=ax.transAxes,
            ha="center",
        )

    fig.subplots_adjust(wspace=-0.045, hspace=0.1)

    # Add the shared colorbars.

    for i, feature in enumerate(["FAPAR", "DD"]):
        cs = peak_color_df[feature].copy().dropna()

        # Separate peak keys into the categories.
        grouped_sorted_peak_keys = []
        for category in ["dd_0", "dd_1", "dd_mix"]:
            for peak_key in cs.index:
                if peak_key in cat_df[feature][category]:
                    grouped_sorted_peak_keys.append(peak_key)

        cs = cs.reindex(grouped_sorted_peak_keys)

        cmap = mpl.colors.ListedColormap(cs)
        bounds = np.arange(0, len(cs) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        box = axes[-1, i].get_position()

        height = 0.02
        y0 = box.ymin - height - 0.034

        width = 0.04 * len(cs)

        x0 = (box.xmin + box.xmax) / 2 - width / 2

        cb = mpl.colorbar.ColorbarBase(
            fig.add_axes([x0, y0, width, height]),
            cmap=cmap,
            boundaries=bounds,
            extend="neither",
            ticks=get_centres(bounds),
            spacing="uniform",
            orientation="horizontal",
            label=f"peak combination ({feature})",
        )
        cb.ax.set_xticklabels(cs.index, rotation=35)
        cb.ax.xaxis.set_label_position("top")

    # Save the combined figure.
    map_figure_saver.save_figure(
        fig,
        f"{experiment.name}_normal_ba_peak_distr_max_shap_FAPAR__DD",
        sub_directory=Path(f"{experiment.name}") / "shap_peaks",
        dpi=350,
    )
