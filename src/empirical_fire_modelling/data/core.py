# -*- coding: utf-8 -*-
"""Creation of the data structures used for fitting."""
import re
from datetime import datetime
from functools import reduce
from pprint import pformat

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from wildfires.analysis import data_processing
from wildfires.data import (
    HYDE,
    VODCA,
    WWLLN,
    AvitabileThurnerAGB,
    Copernicus_SWI,
    Datasets,
    ERA5_DryDayPeriod,
    ERA5_Temperature,
    ESA_CCI_Landcover_PFT,
    GFEDv4,
    GlobFluo_SIF,
    MOD15A2H_LAI_fPAR,
    dataset_times,
)

from ..cache import cache, check_in_store
from ..configuration import (
    experiment_name_dict,
    get_filled_names,
    lags,
    offset_selected_features,
    selected_features,
    st_k,
    st_persistent_perc,
    train_test_split_kwargs,
)

__all__ = ("get_data", "get_experiment_split_data", "get_split_data")


@cache
def _get_processed_data(shift_months=lags[1:]):
    """Low-level function which carries out the basic data processing."""
    target_variable = "GFED4 BA"

    # Variables required for the above.
    required_variables = [target_variable]

    # Dataset selection.

    selection_datasets = [
        AvitabileThurnerAGB(),
        ERA5_Temperature(),
        ESA_CCI_Landcover_PFT(),
        GFEDv4(),
        HYDE(),
        WWLLN(),
    ]

    # Datasets subject to temporal interpolation (filling).
    temporal_interp_datasets = [
        Datasets(Copernicus_SWI()).select_variables(("SWI(1)",)).dataset
    ]

    # Datasets subject to temporal interpolation and shifting.
    shift_and_interp_datasets = [
        Datasets(MOD15A2H_LAI_fPAR()).select_variables(("FAPAR", "LAI")).dataset,
        Datasets(VODCA()).select_variables(("VOD Ku-band",)).dataset,
        Datasets(GlobFluo_SIF()).select_variables(("SIF",)).dataset,
    ]

    # Datasets subject to temporal shifting.
    datasets_to_shift = [
        Datasets(ERA5_DryDayPeriod()).select_variables(("Dry Day Period",)).dataset
    ]

    all_datasets = (
        selection_datasets
        + temporal_interp_datasets
        + shift_and_interp_datasets
        + datasets_to_shift
    )

    # Determine shared temporal extent of the data.
    min_time, max_time = dataset_times(all_datasets)[:2]
    shift_min_time = min_time - relativedelta(years=2)

    # Sanity check.
    assert min_time == datetime(2010, 1, 1)
    assert shift_min_time == datetime(2008, 1, 1)
    assert max_time == datetime(2015, 4, 1)

    for dataset in datasets_to_shift + shift_and_interp_datasets:
        # Apply longer time limit to the datasets to be shifted.
        dataset.limit_months(shift_min_time, max_time)

        for cube in dataset:
            assert cube.shape[0] == 88

    for dataset in selection_datasets + temporal_interp_datasets:
        # Apply time limit.
        dataset.limit_months(min_time, max_time)

        if dataset.frequency == "monthly":
            for cube in dataset:
                assert cube.shape[0] == 64

    for dataset in all_datasets:
        # Regrid each dataset to the common grid.
        dataset.regrid()

    # Calculate and apply the shared mask.
    total_masks = []

    for dataset in temporal_interp_datasets + shift_and_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)

            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > (
                7 * 6  # Up to 6 months for each of the 7 complete years.
                + 10  # Additional Jan, Feb, Mar, Apr, + 6 extra.
            )

            total_masks.append(ignore_mask)

    combined_mask = reduce(np.logical_or, total_masks)

    # Apply mask to all datasets.
    for dataset in all_datasets:
        dataset.apply_masks(combined_mask)

    # Carry out the minima and season-trend filling.
    for datasets in (temporal_interp_datasets, shift_and_interp_datasets):
        for i, dataset in enumerate(datasets):
            datasets[i] = dataset.get_persistent_season_trend_dataset(
                persistent_perc=st_persistent_perc, k=st_k
            )

    datasets_to_shift.extend(shift_and_interp_datasets)
    selection_datasets += datasets_to_shift
    selection_datasets += temporal_interp_datasets

    if shift_months is not None:
        for shift in shift_months:
            for shift_dataset in datasets_to_shift:
                # Remove any temporal coordinates other than 'time' here if needed,
                # since these would otherwise become misaligned when the data is
                # shifted below.
                for cube in shift_dataset:
                    for prune_coord in ("month_number", "year"):
                        if cube.coords(prune_coord):
                            cube.remove_coord(prune_coord)

                selection_datasets.append(
                    shift_dataset.get_temporally_shifted_dataset(
                        months=-shift, deep=False
                    )
                )

    selection_variables = get_filled_names(
        [
            "AGB Tree",
            "Diurnal Temp Range",
            "Dry Day Period",
            "FAPAR",
            "LAI",
            "Max Temp",
            "SIF",
            "SWI(1)",
            "ShrubAll",
            "TreeAll",
            "VOD Ku-band",
            "lightning",
            "pftCrop",
            "pftHerb",
            "popd",
        ]
    )
    if shift_months is not None:
        for shift in shift_months:
            selection_variables.extend(
                [
                    f"{var} {-shift} Month"
                    for var in get_filled_names(
                        ["LAI", "FAPAR", "Dry Day Period", "VOD Ku-band", "SIF"]
                    )
                ]
            )

    selection_variables = list(set(selection_variables).union(required_variables))

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which="climatology",
        transformations={},
        deletions=[],
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
        masks=None,
    )
    return (
        # XXX: testing
        endog_data.iloc[:10000],
        exog_data.iloc[:10000],
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@cache
def _get_offset_data(
    endog_data,
    exog_data,
    master_mask,
    filled_datasets,
    masked_datasets,
    land_mask,
):
    """Low-level function which calculates anomalies for large lags.

    The arguments:
        `endog_data, exog_data, master_mask, filled_datasets, masked_datasets,
        land_mask`
    are output by `_get_processed_data()`.

    """
    to_delete = []

    for column in exog_data:
        match = re.search(r"-\d{1,2}", column)
        if match:
            span = match.span()
            # Change the string to reflect the shift.
            original_offset = int(column[slice(*span)])
            if original_offset > -12:
                # Only shift months that are 12 or more months before the current month.
                continue
            comp = -(-original_offset % 12)
            new_column = " ".join(
                (
                    column[: span[0] - 1],
                    f"{original_offset} - {comp}",
                    column[span[1] + 1 :],
                )
            )
            if comp == 0:
                comp_column = column[: span[0] - 1]
            else:
                comp_column = " ".join(
                    (column[: span[0] - 1], f"{comp}", column[span[1] + 1 :])
                )
            print(column, comp_column)
            exog_data[new_column] = exog_data[column] - exog_data[comp_column]
            to_delete.append(column)

    for column in to_delete:
        del exog_data[column]

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


def get_data(experiment="ALL", cache_check=False):
    """Get data for a given experiment."""
    experiment = experiment_name_dict.get(experiment, experiment)
    if experiment not in experiment_name_dict.values():
        name_dict_str = pformat(experiment_name_dict)
        raise ValueError(
            f"The given experiment '{experiment}' was not found in:\n{name_dict_str}."
        )
    if cache_check:
        check_in_store(_get_processed_data)
        return check_in_store(_get_offset_data, *_get_processed_data())
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = _get_offset_data(*_get_processed_data())

    exp_selected_features = selected_features[experiment]
    exp_offset_selected_features = offset_selected_features[experiment]
    if exp_offset_selected_features is not None:
        assert len(exp_selected_features) == len(exp_offset_selected_features) == 15
        # We need to subset exog_data, filled_datasets, and masked_datasets.
        exog_data = exog_data[list(exp_offset_selected_features)]
        filled_datasets = filled_datasets.select_variables(exp_selected_features)
        masked_datasets = masked_datasets.select_variables(exp_selected_features)
        assert (
            exog_data.shape[1]
            == len(filled_datasets.cubes)
            == len(masked_datasets.cubes)
            == 15
        )

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@cache
def get_split_data(
    exog_data, endog_data, train_test_split_kwargs=train_test_split_kwargs
):
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data, endog_data, **train_test_split_kwargs
    )
    return X_train, X_test, y_train, y_test


def get_experiment_split_data(experiment, cache_check=False):
    if cache_check:
        check_in_store(get_data, experiment=experiment)
    endog_data, exog_data = get_data(experiment=experiment)[:2]
    return get_split_data(exog_data, endog_data, cache_check=cache_check)
