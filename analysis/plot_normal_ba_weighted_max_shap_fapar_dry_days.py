# -*- coding: utf-8 -*-
"""15VEG_FAPAR SHAP map plotting."""
import logging
import sys
import warnings
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import numpy as np
from loguru import logger as loguru_logger
from wildfires.data import dummy_lat_lon_cube
from wildfires.utils import get_masked_array, shorten_features

from empirical_fire_modelling import variable
from empirical_fire_modelling.analysis.shap import (
    calculate_2d_masked_shap_values,
    get_max_positions,
    get_shap_values,
)
from empirical_fire_modelling.configuration import Experiment
from empirical_fire_modelling.data import get_endog_exog_mask, get_experiment_split_data
from empirical_fire_modelling.logging_config import enable_logging
from empirical_fire_modelling.model import get_model
from empirical_fire_modelling.plotting import (
    SetupFourMapAxes,
    disc_cube_plot,
    get_aux0_aux1_kwargs,
    map_figure_saver,
)
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

    def param_iter():
        for variable_factory in tqdm(
            [variable.FAPAR, variable.DRY_DAY_PERIOD], desc="Feature"
        ):
            for exclude_inst in tqdm([False, True], desc="Exclude inst."):
                yield exclude_inst, variable_factory

    weighted_plot_data = {}
    for exclude_inst, variable_factory in param_iter():
        weighted_plot_data[(exclude_inst, variable_factory)] = get_max_positions(
            X=X_test,
            variables=[variable_factory[lag] for lag in chosen_lags],
            shap_results=map_shap_results,
            shap_measure="masked_max_shap_arrs",
            mean_ba=mean_ba,
            exclude_inst=exclude_inst,
            ptp_threshold_factor=ptp_threshold_factor,
            diff_threshold=diff_threshold,
        )

    with SetupFourMapAxes(cbar="horizontal") as (fig, axes, cax):
        for (
            i,
            (ax, ((exclude_inst, variable_factory), max_positions), title),
        ) in enumerate(
            zip(
                axes,
                weighted_plot_data.items(),
                ascii_lowercase[: len(axes)],
            )
        ):
            # For some reason, the functions involved in plotting don't behave well
            # with HashProxy instances, so just grab the underlying data here (which
            # is done later on anyway).
            max_positions = max_positions.__wrapped__
            disc_cube_plot(
                dummy_lat_lon_cube(max_positions),
                ax=ax,
                bin_edges=np.linspace(1, 6, 11),
                extend="both",
                cmap="viridis",
                cbar=i == 3,
                cbar_label="month",
                cbar_format="%0.0f",
                cax=cax,
                cbar_orientation="horizontal",
                **get_aux0_aux1_kwargs(y_test, master_mask),
                loc=(0.75, 0.11),
                height=0.032,
                aspect=1.35,
            )
            short_feature = shorten_features(str(variable_factory))
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

    # Save the combined figure.
    map_figure_saver.save_figure(
        fig,
        f"{experiment.name}_normal_ba_weighted_max_shap_FAPAR__DD",
        sub_directory=Path(f"{experiment.name}") / "weighted_shap_maps",
        dpi=350,
    )
