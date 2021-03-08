# -*- coding: utf-8 -*-
"""Creation of the data structures used for fitting."""
from datetime import datetime
from functools import reduce
from operator import attrgetter, methodcaller

import numpy as np
import pandas as pd
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

from .. import variable
from ..cache import cache, mark_dependency
from ..configuration import Experiment, selected_features, train_test_split_kwargs

__all__ = ("get_data", "get_experiment_split_data", "get_split_data")


@cache
@mark_dependency
def _get_processed_data(
    shift_months=variable.lags[1:],
    persistent_perc=variable.st_persistent_perc,
    season_trend_k=variable.st_k,
    all_inst_features=selected_features[Experiment.CURR],
    shifted_variables=variable.shifted_variables,
    _variable_names=tuple(
        map(
            attrgetter("name", "shift"),
            selected_features[Experiment.ALL],
        )
    ),
):
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
        Datasets(Copernicus_SWI()).select_variables((variable.SWI.name,)).dataset
    ]

    # Datasets subject to temporal interpolation and shifting.
    shift_and_interp_datasets = [
        Datasets(MOD15A2H_LAI_fPAR())
        .select_variables((variable.FAPAR.name, variable.LAI.name))
        .dataset,
        Datasets(VODCA()).select_variables((variable.VOD.name,)).dataset,
        Datasets(GlobFluo_SIF()).select_variables((variable.SIF.name,)).dataset,
    ]

    # Datasets subject to temporal shifting.
    datasets_to_shift = [
        Datasets(ERA5_DryDayPeriod())
        .select_variables((variable.DRY_DAY_PERIOD.name,))
        .dataset
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
                persistent_perc=persistent_perc, k=season_trend_k
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

    selection_variables = list(map(attrgetter("raw_filled"), all_inst_features))
    if shift_months is not None:
        for shift in shift_months:
            for var_factory in shifted_variables:
                selection_variables.append(var_factory[shift].raw_filled)

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

    def _pandas_string_labels_to_variables(x):
        """Transform series names or columns labels to variable.Variable instances."""
        all_variables = tuple(
            # Get the instantaneous variables corresponding to all variables.
            list(map(methodcaller("get_standard"), selected_features[Experiment.ALL]))
            + [variable.GFED4_BA]
        )
        all_variable_names = tuple(map(attrgetter("raw_filled"), all_variables))
        if isinstance(x, pd.Series):
            x.name = all_variables[all_variable_names.index(x.name)]
        elif isinstance(x, pd.DataFrame):
            x.columns = [all_variables[all_variable_names.index(c)] for c in x.columns]
        else:
            raise TypeError(
                f"Expected either a pandas.Series or pandas.DataFrame. Got '{x}'."
            )

    _pandas_string_labels_to_variables(endog_data)
    _pandas_string_labels_to_variables(exog_data)

    assert exog_data.shape[1] == 50

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@cache
@mark_dependency
def _get_offset_exog_data(
    exog_data,
):
    """Low-level function which calculates anomalies for large lags."""
    to_delete = []

    for var in exog_data:
        if var.shift < 12:
            continue

        new_var = var.get_offset()
        comp_var = variable.get_matching(
            exog_data.columns, name=new_var.name, shift=new_var.comp_shift
        )
        print(f"{var} - {comp_var} -> {new_var}")
        exog_data[new_var] = exog_data[var] - exog_data[comp_var]
        to_delete.append(var)

    for column in to_delete:
        del exog_data[column]

    return exog_data


@cache(dependencies=(_get_processed_data, _get_offset_exog_data))
@mark_dependency
def get_data(experiment="ALL"):
    """Get data for a given experiment."""
    (
        endog_data,
        unshifted_exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = _get_processed_data()

    exog_data = _get_offset_exog_data(unshifted_exog_data)

    # Since we applied offsets above, this needs to be reflected in the variable names.
    exp_selected_features = tuple(
        map(methodcaller("get_offset"), selected_features[experiment])
    )
    if set(exp_selected_features) != set(exog_data.columns):
        assert len(exp_selected_features) == 15
        # We need to subset exog_data, filled_datasets, and masked_datasets.
        exog_data = exog_data[list(exp_selected_features)]
        # The Datasets objects below are not ware of the 'variable' module and use
        # normal string indexing instead.
        filled_datasets = filled_datasets.select_variables(
            tuple(map(attrgetter("raw_filled"), exp_selected_features))
        )
        masked_datasets = masked_datasets.select_variables(
            tuple(map(attrgetter("raw_filled"), exp_selected_features))
        )
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
@mark_dependency
def get_split_data(
    exog_data, endog_data, train_test_split_kwargs=train_test_split_kwargs
):
    X_train, X_test, y_train, y_test = train_test_split(
        exog_data, endog_data, **train_test_split_kwargs
    )
    return X_train, X_test, y_train, y_test


@cache(dependencies=(get_split_data, get_data))
def get_experiment_split_data(experiment):
    endog_data, exog_data = get_data(experiment=experiment)[:2]
    return get_split_data(exog_data, endog_data)
