# -*- coding: utf-8 -*-
"""Creation of the data structures used for fitting."""
from collections import Counter
from datetime import datetime
from functools import reduce
from operator import attrgetter, methodcaller

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from immutabledict import immutabledict
from iris.time import PartialDateTime
from sklearn.model_selection import train_test_split
from wildfires.analysis import data_processing
from wildfires.data import MCD64CMQ_C6, Dataset, Datasets, GFEDv4, dataset_times

from .. import variable
from ..cache import cache, mark_dependency, memory, process_proxy
from ..configuration import (
    Experiment,
    Filling,
    selected_features,
    train_test_split_kwargs,
)

__all__ = (
    "ba_dataset_map",
    "get_data",
    "get_endog_exog_mask",
    "get_experiment_split_data",
    "get_first_cube_datetimes",
    "get_map_data",
    "get_split_data",
    "get_frac_train_nr_samples",
)


ba_dataset_map = immutabledict(
    {
        variable.MCD64CMQ_BA: MCD64CMQ_C6,
        variable.GFED4_BA: GFEDv4,
    }
)


@cache
@mark_dependency
def _nn_basis_func(
    *,
    check_max_time,
    check_min_time,
    check_shift_min_time,
    exp_features,
    normal_mask_ignore_n,
    shift_mask_ignore_n,
    max_time=None,
    min_time=None,
    n_months,
    normal_n_time,
    shift_n_time,
    spec_datasets_to_shift,
    spec_selection_datasets,
    spec_shift_and_interp_datasets,
    spec_temporal_interp_datasets,
    target_var,
    which,
    all_shifted_variables=variable.shifted_variables,
    # Store this initially, since this is changed as new datasets (e.g. filled
    # datasets) are derived from the original datasets.
    original_datasets=tuple(sorted(Dataset.datasets, key=attrgetter("__name__"))),
):
    target_variable = target_var.name

    required_variables = [target_variable]

    shifted_variables = {var.parent for var in exp_features if var.shift != 0}
    assert all(
        shifted_var in all_shifted_variables for shifted_var in shifted_variables
    )

    shift_months = [
        shift for shift in sorted({var.shift for var in exp_features}) if shift != 0
    ]

    def year_month_datetime(dt):
        """Use only year and month information to construct a datetime."""
        return datetime(dt.year, dt.month, 1)

    def create_dataset_group(spec):
        """Create a dataset group from its specification."""
        group = []
        for dataset_name, selected_variables in spec.items():
            # Select the relevant dataset.
            matching_datasets = [
                d for d in original_datasets if d.__name__ == dataset_name
            ]
            if not len(matching_datasets) == 1:
                raise ValueError(
                    f"Expected 1 matching dataset for '{dataset_name}', "
                    f"got {matching_datasets}."
                )
            # Instantiate the matching Dataset.
            matching_dataset = matching_datasets[0]()
            if selected_variables:
                # There are variables to select.
                group.append(
                    Datasets(matching_dataset)
                    .select_variables(selected_variables)
                    .dataset
                )
            else:
                # There is nothing to select.
                group.append(matching_dataset)
        return group

    selection_datasets = create_dataset_group(spec_selection_datasets)
    temporal_interp_datasets = create_dataset_group(spec_temporal_interp_datasets)
    shift_and_interp_datasets = create_dataset_group(spec_shift_and_interp_datasets)
    datasets_to_shift = create_dataset_group(spec_datasets_to_shift)

    all_datasets = (
        selection_datasets
        + temporal_interp_datasets
        + shift_and_interp_datasets
        + datasets_to_shift
    )

    # Determine shared temporal extent of the data.
    _min_time, _max_time, _times_df = dataset_times(all_datasets)

    print(_times_df)

    if min_time is None:
        min_time = _min_time
    if max_time is None:
        max_time = _max_time

    assert min_time >= _min_time
    assert max_time <= _max_time

    if shift_months:
        _shift_min_time = year_month_datetime(min_time) - relativedelta(
            months=shift_months[-1]
        )
        shift_min_time = PartialDateTime(
            year=_shift_min_time.year, month=_shift_min_time.month
        )
    else:
        shift_min_time = min_time

    # Sanity check.
    assert min_time == check_min_time
    assert shift_min_time == check_shift_min_time
    assert max_time == check_max_time

    for dataset in datasets_to_shift:
        # Apply longer time limit to the datasets to be shifted.
        dataset.limit_months(shift_min_time, max_time)

        for cube in dataset:
            assert cube.shape[0] == shift_n_time

    for dataset in selection_datasets:
        # Apply time limit.
        dataset.limit_months(min_time, max_time)

        if dataset.frequency == "monthly":
            for cube in dataset:
                assert cube.shape[0] == normal_n_time

    for dataset in shift_and_interp_datasets:
        # Apply longer time limit to the datasets to be shifted.
        dataset.limit_months(
            year_month_datetime(shift_min_time) - relativedelta(months=+n_months),
            year_month_datetime(max_time) + relativedelta(months=+n_months),
        )

        for cube in dataset:
            assert cube.shape[0] == shift_n_time + 2 * n_months

    for dataset in temporal_interp_datasets:
        # Apply time limit.
        dataset.limit_months(
            year_month_datetime(min_time) - relativedelta(months=+n_months),
            year_month_datetime(max_time) + relativedelta(months=+n_months),
        )

        if dataset.frequency == "monthly":
            for cube in dataset:
                assert cube.shape[0] == normal_n_time + 2 * n_months

    for dataset in all_datasets:
        # Regrid each dataset to the common grid.
        dataset.regrid()

    # Calculate and apply the shared mask.
    total_masks = []

    for dataset in temporal_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)
            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > normal_mask_ignore_n
            total_masks.append(ignore_mask)

    for dataset in shift_and_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)
            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > shift_mask_ignore_n
            total_masks.append(ignore_mask)

    combined_mask = reduce(np.logical_or, total_masks)

    # Apply mask to all datasets.
    for dataset in all_datasets:
        dataset.apply_masks(combined_mask)

    # Carry out the nearest-neighbour filling.
    for i, dataset in enumerate(temporal_interp_datasets):
        temporal_interp_datasets[i] = dataset.get_temporally_interpolated_dataset(
            target_timespan=tuple(map(year_month_datetime, (min_time, max_time))),
            n_months=n_months,
            verbose=True,
        )
    for i, dataset in enumerate(shift_and_interp_datasets):
        shift_and_interp_datasets[i] = dataset.get_temporally_interpolated_dataset(
            target_timespan=tuple(map(year_month_datetime, (shift_min_time, max_time))),
            n_months=n_months,
            verbose=True,
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

    selection_variables = list(
        set(map(lambda v: v.get_standard().raw_nn_filled, exp_features)).union(
            required_variables
        )
    )

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        _,  # We don't need the `filled_datasets`.
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which=which,
        transformations={},
        deletions=[],
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
        masks=None,
    )

    def _pandas_string_labels_to_variables(
        x,
        target_var,
        all_features=selected_features[Experiment.ALL],
    ):
        """Transform series names or columns labels to variable.Variable instances."""

        all_variables = tuple(
            # Get the instantaneous variables corresponding to all variables.
            list(map(methodcaller("get_standard"), all_features))
            + [target_var]
        )
        all_variable_names = tuple(map(attrgetter("raw_nn_filled"), all_variables))
        if isinstance(x, pd.Series):
            x.name = all_variables[all_variable_names.index(x.name)]
        elif isinstance(x, pd.DataFrame):
            x.columns = [all_variables[all_variable_names.index(c)] for c in x.columns]
        else:
            raise TypeError(
                f"Expected either a pandas.Series or pandas.DataFrame. Got '{x}'."
            )

    _pandas_string_labels_to_variables(endog_data, target_var)
    _pandas_string_labels_to_variables(exog_data, target_var)

    assert exog_data.shape[1] == len(exp_features)

    # Calculate anomalies for large lags.
    # NOTE: Modifies `exog_data` inplace.
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

    # Check again.
    assert exog_data.shape[1] == len(exp_features)

    return (
        endog_data,
        exog_data,
        master_mask,
        masked_datasets,
        land_mask,
        set(exog_data.columns),
    )


@cache
@mark_dependency
def _basis_func(
    *,
    check_max_time,
    check_min_time,
    check_shift_min_time,
    exp_features,
    normal_mask_ignore_n,
    shift_mask_ignore_n,
    max_time=None,
    min_time=None,
    normal_n_time,
    persistent_perc,
    season_trend_k,
    shift_n_time,
    spec_datasets_to_shift,
    spec_selection_datasets,
    spec_shift_and_interp_datasets,
    spec_temporal_interp_datasets,
    target_var,
    which,
    all_shifted_variables=variable.shifted_variables,
    # Store this initially, since this is changed as new datasets (e.g. filled
    # datasets) are derived from the original datasets.
    original_datasets=tuple(sorted(Dataset.datasets, key=attrgetter("__name__"))),
):
    target_variable = target_var.name

    required_variables = [target_variable]

    shifted_variables = {var.parent for var in exp_features if var.shift != 0}
    assert all(
        shifted_var in all_shifted_variables for shifted_var in shifted_variables
    )

    shift_months = [
        shift for shift in sorted({var.shift for var in exp_features}) if shift != 0
    ]

    def create_dataset_group(spec):
        """Create a dataset group from its specification."""
        group = []
        for dataset_name, selected_variables in spec.items():
            # Select the relevant dataset.
            matching_datasets = [
                d for d in original_datasets if d.__name__ == dataset_name
            ]
            if not len(matching_datasets) == 1:
                raise ValueError(
                    f"Expected 1 matching dataset for '{dataset_name}', "
                    f"got {matching_datasets}."
                )
            # Instantiate the matching Dataset.
            matching_dataset = matching_datasets[0]()
            if selected_variables:
                # There are variables to select.
                group.append(
                    Datasets(matching_dataset)
                    .select_variables(selected_variables)
                    .dataset
                )
            else:
                # There is nothing to select.
                group.append(matching_dataset)
        return group

    selection_datasets = create_dataset_group(spec_selection_datasets)
    temporal_interp_datasets = create_dataset_group(spec_temporal_interp_datasets)
    shift_and_interp_datasets = create_dataset_group(spec_shift_and_interp_datasets)
    datasets_to_shift = create_dataset_group(spec_datasets_to_shift)

    all_datasets = (
        selection_datasets
        + temporal_interp_datasets
        + shift_and_interp_datasets
        + datasets_to_shift
    )

    # Determine shared temporal extent of the data.
    _min_time, _max_time = dataset_times(all_datasets)[:2]

    if min_time is None:
        min_time = _min_time
    if max_time is None:
        max_time = _max_time

    assert min_time >= _min_time
    assert max_time <= _max_time

    if shift_months:
        _shift_min_time = datetime(min_time.year, min_time.month, 1) - relativedelta(
            months=shift_months[-1]
        )
        shift_min_time = PartialDateTime(
            year=_shift_min_time.year, month=_shift_min_time.month
        )
    else:
        shift_min_time = min_time

    # Sanity check.
    assert min_time == check_min_time
    assert shift_min_time == check_shift_min_time
    assert max_time == check_max_time

    for dataset in datasets_to_shift + shift_and_interp_datasets:
        # Apply longer time limit to the datasets to be shifted.
        dataset.limit_months(shift_min_time, max_time)

        for cube in dataset:
            assert cube.shape[0] == shift_n_time

    for dataset in selection_datasets + temporal_interp_datasets:
        # Apply time limit.
        dataset.limit_months(min_time, max_time)

        if dataset.frequency == "monthly":
            for cube in dataset:
                assert cube.shape[0] == normal_n_time

    for dataset in all_datasets:
        # Regrid each dataset to the common grid.
        dataset.regrid()

    # Calculate and apply the shared mask.
    total_masks = []

    for dataset in temporal_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)
            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > normal_mask_ignore_n
            total_masks.append(ignore_mask)

    for dataset in shift_and_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)
            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > shift_mask_ignore_n
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

    selection_variables = list(
        set(map(lambda v: v.get_standard().raw_filled, exp_features)).union(
            required_variables
        )
    )

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        _,  # We don't need the `filled_datasets`.
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which=which,
        transformations={},
        deletions=[],
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
        masks=None,
    )

    def _pandas_string_labels_to_variables(
        x,
        target_var,
        all_features=selected_features[Experiment.ALL],
    ):
        """Transform series names or columns labels to variable.Variable instances."""

        all_variables = tuple(
            # Get the instantaneous variables corresponding to all variables.
            list(map(methodcaller("get_standard"), all_features))
            + [target_var]
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

    _pandas_string_labels_to_variables(endog_data, target_var)
    _pandas_string_labels_to_variables(exog_data, target_var)

    assert exog_data.shape[1] == len(exp_features)

    # Calculate anomalies for large lags.
    # NOTE: Modifies `exog_data` inplace.
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

    # Check again.
    assert exog_data.shape[1] == len(exp_features)

    return (
        endog_data,
        exog_data,
        master_mask,
        masked_datasets,
        land_mask,
        set(exog_data.columns),
    )


def get_ignore_n(start, end):
    start = datetime(year=start.year, month=start.month, day=1)
    end = datetime(year=end.year, month=end.month, day=1)
    if end < start:
        raise ValueError("end >= start was not fulfilled")
    month_numbers = []
    while start != end:
        month_numbers.append(start.month)
        start += relativedelta(months=+1)
    month_numbers.append(start.month)
    counts = Counter(month_numbers)
    return (
        counts[10]
        + counts[11]
        + counts[12]
        + counts[1]
        + counts[2]
        + counts[3]
        + len(month_numbers) // 12
    )


@mark_dependency
def get_data(
    experiment=Experiment.ALL,
    persistent_perc=variable.st_persistent_perc,
    season_trend_k=variable.st_k,
    nn_n_months=variable.nn_n_months,
    all_features=selected_features,
    _variable_names=tuple(
        map(
            attrgetter("name", "shift"),
            selected_features[Experiment.ALL],
        )
    ),
    cache_check=False,
):
    """Get data for a given experiment."""
    if experiment == Experiment["15VEG_FAPAR_MON"]:
        filling = Filling.ST
        target_var = variable.MCD64CMQ_BA
        exp_features = all_features[Experiment["15VEG_FAPAR_MON"]]
        which = "monthly"

        # Dataset selection.

        selection_datasets = {
            "AvitabileThurnerAGB": set(),
            "ERA5_Temperature": set(),
            "Ext_ESA_CCI_Landcover_PFT": set(),
            "HYDE": set(),
            ba_dataset_map[target_var].__name__: set(),
        }

        # Datasets subject to temporal interpolation (filling).
        temporal_interp_datasets = {}

        # Datasets subject to temporal interpolation and shifting.
        shift_and_interp_datasets = {
            "Ext_MOD15A2H_fPAR": {
                variable.FAPAR.name,
            }
        }

        # Datasets subject to temporal shifting.
        datasets_to_shift = {
            "ERA5_DryDayPeriod": {
                variable.DRY_DAY_PERIOD.name,
            }
        }

        # The FAPAR dataset begins 2000-02.
        min_time = PartialDateTime(year=2000, month=11)

        # The ESA CCI LC dataset stops in 2019.
        max_time = PartialDateTime(year=2019, month=12)

        # Sanity check including shifting.
        check_min_time = PartialDateTime(year=2000, month=11)
        check_shift_min_time = PartialDateTime(year=2000, month=2)
        check_max_time = PartialDateTime(year=2019, month=12)

        shift_n_time = 239
        normal_n_time = 230

        normal_mask_ignore_n = get_ignore_n(check_min_time, check_max_time)
        assert normal_mask_ignore_n == 19 * 6 + 2 + 19

        shift_mask_ignore_n = get_ignore_n(check_shift_min_time, check_max_time)
        assert shift_mask_ignore_n == 19 * 6 + 5 + 19
    elif experiment == Experiment.ALL_NN:
        filling = Filling.NN
        target_var = variable.GFED4_BA
        exp_features = all_features[Experiment.ALL]
        which = "climatology"

        # Dataset selection.
        selection_datasets = {
            "AvitabileThurnerAGB": set(),
            "ERA5_Temperature": set(),
            "Ext_ESA_CCI_Landcover_PFT": set(),
            "HYDE": set(),
            "WWLLN": set(),
            ba_dataset_map[target_var].__name__: set(),
        }

        # Datasets subject to temporal interpolation (filling).
        temporal_interp_datasets = {
            "Copernicus_SWI": {
                variable.SWI.name,
            },
        }

        # Datasets subject to temporal interpolation and shifting.
        shift_and_interp_datasets = {
            "MOD15A2H_LAI_fPAR": {
                variable.LAI.name,
            },
            "Ext_MOD15A2H_fPAR": {
                variable.FAPAR.name,
            },
            "VODCA": {
                variable.VOD.name,
            },
            "GlobFluo_SIF": {
                variable.SIF.name,
            },
        }

        # Datasets subject to temporal shifting.
        datasets_to_shift = {
            "ERA5_DryDayPeriod": {
                variable.DRY_DAY_PERIOD.name,
            }
        }

        min_time = PartialDateTime(year=2010, month=1)
        # 3-months at the end are reserved for NN interpolation
        max_time = PartialDateTime(year=2015, month=1)

        # Sanity check.
        check_min_time = PartialDateTime(year=2010, month=1)
        check_shift_min_time = PartialDateTime(year=2008, month=1)
        check_max_time = PartialDateTime(year=2015, month=1)

        shift_n_time = 85
        normal_n_time = 61

        normal_mask_ignore_n = get_ignore_n(check_min_time, check_max_time)
        assert normal_mask_ignore_n == 5 * 6 + 6

        shift_mask_ignore_n = get_ignore_n(check_shift_min_time, check_max_time)
        assert shift_mask_ignore_n == 7 * 6 + 8
    else:
        filling = Filling.ST
        target_var = variable.GFED4_BA
        exp_features = all_features[Experiment.ALL]
        which = "climatology"

        # Dataset selection.
        selection_datasets = {
            "AvitabileThurnerAGB": set(),
            "ERA5_Temperature": set(),
            "Ext_ESA_CCI_Landcover_PFT": set(),
            "HYDE": set(),
            "WWLLN": set(),
            ba_dataset_map[target_var].__name__: set(),
        }

        # Datasets subject to temporal interpolation (filling).
        temporal_interp_datasets = {
            "Copernicus_SWI": {
                variable.SWI.name,
            },
        }

        # Datasets subject to temporal interpolation and shifting.
        shift_and_interp_datasets = {
            "MOD15A2H_LAI_fPAR": {
                variable.LAI.name,
            },
            "Ext_MOD15A2H_fPAR": {
                variable.FAPAR.name,
            },
            "VODCA": {
                variable.VOD.name,
            },
            "GlobFluo_SIF": {
                variable.SIF.name,
            },
        }

        # Datasets subject to temporal shifting.
        datasets_to_shift = {
            "ERA5_DryDayPeriod": {
                variable.DRY_DAY_PERIOD.name,
            }
        }

        # Data-derived.
        min_time = None
        max_time = None

        # Sanity check.
        check_min_time = PartialDateTime(year=2010, month=1)
        check_shift_min_time = PartialDateTime(year=2008, month=1)
        check_max_time = PartialDateTime(year=2015, month=4)

        shift_n_time = 88
        normal_n_time = 64

        normal_mask_ignore_n = get_ignore_n(check_min_time, check_max_time)
        assert normal_mask_ignore_n == 5 * 6 + 8

        shift_mask_ignore_n = get_ignore_n(check_shift_min_time, check_max_time)
        assert shift_mask_ignore_n == 7 * 6 + 10

    # Actually retrieve the specified data.
    data_kwargs = dict(
        check_max_time=check_max_time,
        check_min_time=check_min_time,
        check_shift_min_time=check_shift_min_time,
        exp_features=exp_features,
        normal_mask_ignore_n=normal_mask_ignore_n,
        shift_mask_ignore_n=shift_mask_ignore_n,
        max_time=max_time,
        min_time=min_time,
        normal_n_time=normal_n_time,
        shift_n_time=shift_n_time,
        spec_datasets_to_shift=datasets_to_shift,
        spec_selection_datasets=selection_datasets,
        spec_shift_and_interp_datasets=shift_and_interp_datasets,
        spec_temporal_interp_datasets=temporal_interp_datasets,
        target_var=target_var,
        which=which,
    )

    if filling == Filling.ST:
        data_kwargs.update(
            dict(
                persistent_perc=persistent_perc,
                season_trend_k=season_trend_k,
            )
        )
        data_basis_func = _basis_func
    elif filling == Filling.NN:
        data_kwargs.update(
            dict(
                n_months=nn_n_months,
            )
        )
        data_basis_func = _nn_basis_func
    else:
        raise ValueError("Unsupported filling.")

    if cache_check:
        return data_basis_func.check_in_store(**data_kwargs)

    (
        endog_data,
        exog_data,
        master_mask,
        masked_datasets,
        land_mask,
        exog_data_columns,
    ) = data_basis_func(**data_kwargs)

    # Since we applied offsets above, this needs to be reflected in the variable names.
    exp_selected_features = tuple(
        map(methodcaller("get_offset"), all_features[experiment])
    )

    if memory.get_hash(set(exp_selected_features)) != memory.get_hash(
        exog_data_columns
    ):
        assert len(exp_selected_features) == 15

        # We need to subset exog_data and masked_datasets.

        # Do this lazily to avoid realising the cached data.

        def select_exog_data(df, selection=tuple(exp_selected_features)):
            df = df[list(selection)]
            assert df.shape[1] == 15
            return df

        def select_masked_datasets(ds, selection=tuple(exp_selected_features)):
            # The Datasets objects below are not ware of the 'variable' module and use
            # normal string indexing instead.
            ds = ds.select_variables(tuple(map(attrgetter("raw_filled"), selection)))
            assert len(ds.cubes) == 15
            return ds

        exog_data = process_proxy((exog_data,), (select_exog_data,))[0]
        masked_datasets = process_proxy((masked_datasets,), (select_masked_datasets,))[
            0
        ]

    return (
        endog_data,
        exog_data,
        master_mask,
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


@cache(dependencies=(get_split_data, get_data, _basis_func))
@mark_dependency
def get_experiment_split_data(experiment):
    endog_data, exog_data = get_endog_exog_mask(experiment=experiment)[:2]
    return get_split_data(exog_data, endog_data)


@mark_dependency
def get_map_data(data_1d, master_mask):
    """Go from 1D data to data on a map, defined by master_mask."""
    map_data = np.ma.MaskedArray(
        np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
    )
    map_data[~master_mask] = data_1d
    return map_data


@cache(dependencies=(get_data, _basis_func))
def get_endog_exog_mask(experiment):
    endog_data, exog_data, master_mask = get_data(experiment=experiment)[:3]
    return endog_data, exog_data, master_mask


@cache
def get_first_cube_datetimes(datasets):
    """Given a Datasets instance, get the datetimes associated with the first cube."""
    datetimes = [
        PartialDateTime(
            year=datasets[0].cubes[0].coord("time").cell(i).point.year,
            month=datasets[0].cubes[0].coord("time").cell(i).point.month,
        )
        for i in range(datasets[0].cubes[0].shape[0])
    ]
    return datetimes


@cache(dependencies=(get_experiment_split_data, get_split_data, get_data, _basis_func))
def get_frac_train_nr_samples(experiment, fraction):
    """Return the number of samples corresponding to a given fraction of the train set."""
    X_train = get_experiment_split_data(experiment)[0]
    return round(fraction * X_train.shape[0])
