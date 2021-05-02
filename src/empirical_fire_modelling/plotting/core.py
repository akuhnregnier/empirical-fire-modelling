# -*- coding: utf-8 -*-
from itertools import product
from pathlib import Path
from string import ascii_lowercase

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from wildfires.analysis import FigureSaver
from wildfires.analysis import cube_plotting as orig_cube_plotting
from wildfires.data import dummy_lat_lon_cube
from wildfires.utils import (
    get_land_mask,
    shorten_features,
    simple_sci_format,
    update_nested_dict,
)

from ..cache import cache
from ..configuration import (
    figure_save_dir,
    figure_saver_kwargs,
    map_figure_saver_kwargs,
)
from ..data import ba_dataset_map
from .spec_cube_plot import disc_cube_plot
from .utils import get_sci_format

__all__ = (
    "SetupFourMapAxes",
    "ba_plotting",
    "cube_plotting",
    "figure_saver",
    "get_aux0_aux1_kwargs",
    "map_figure_saver",
    "plot_shap_value_maps",
)


def cube_plotting(*args, **kwargs):
    """Modified cube plotting with default arguments."""
    kwargs = kwargs.copy()
    assert len(args) <= 1, "At most 1 positional argument supported."
    # Assume certain default kwargs, unless overriden.
    cbar_fmt = ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))
    defaults = dict(
        coastline_kwargs=dict(linewidth=0.3),
        gridline_kwargs=dict(zorder=0, alpha=0.8, linestyle="--", linewidth=0.3),
        colorbar_kwargs=dict(format=cbar_fmt, pad=0.02),
    )
    kwargs = update_nested_dict(defaults, kwargs)
    return orig_cube_plotting(*args, **kwargs)


figure_saver = FigureSaver(directories=figure_save_dir, **figure_saver_kwargs)
map_figure_saver = FigureSaver(directories=figure_save_dir, **map_figure_saver_kwargs)


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
            colorbar_kwargs={
                "format": "%0.1e",
                "label": f"SHAP ('{shorten_features(feature)}')",
            },
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


def ba_plotting(
    predicted_ba,
    masked_val_data,
    figure_saver,
    cbar_label_x_offset=None,
    aux0=None,
    aux1=None,
    aux0_label="",
    aux1_label="",
    filename=None,
):
    # date_str = "2010-01 to 2015-04"
    text_xy = (0.02, 0.935)

    fig, axes = plt.subplots(
        3,
        1,
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(5.1, (2.3 + 0.01) * 3),
        gridspec_kw={"hspace": 0.01, "wspace": 0.01},
    )

    if not np.all(predicted_ba.mask == masked_val_data.mask):
        raise ValueError("Predicted BA and Val BA mask should match.")

    boundaries = np.geomspace(1e-5, 1e-1, 8)

    cmap = "inferno_r"
    extend = "both"

    plot_kwargs = dict(
        aux0=aux0,
        aux1=aux1,
        fig=fig,
        # Disable checking of the displayed bin edges, since we are sure about which
        # bin edges we wnat to display (the ones calculated above).
        cbar_format=get_sci_format(ndigits=1, atol=np.inf),
        cbar_pad=0.035,
    )

    # Plotting observed.
    fig, _, cb0 = disc_cube_plot(
        dummy_lat_lon_cube(np.mean(masked_val_data, axis=0)),
        bin_edges=boundaries,
        ax=axes[0],
        cmap=cmap,
        extend=extend,
        cbar_label="Ob. BA",
        **plot_kwargs,
    )

    # Plotting predicted.
    fig, _, cb1 = disc_cube_plot(
        dummy_lat_lon_cube(np.mean(predicted_ba, axis=0)),
        bin_edges=boundaries,
        ax=axes[1],
        cmap=cmap,
        extend=extend,
        cbar_label="Pr. BA",
        **plot_kwargs,
    )

    frac_diffs = np.mean(masked_val_data - predicted_ba, axis=0) / np.mean(
        masked_val_data, axis=0
    )

    # Plotting differences.
    diff_boundaries = [
        -(10 ** 2),
        -(10 ** 1),
        -(10 ** 0),
        -(3 * 10 ** -2),
        0,
        3 * 10 ** -2,
        3 * 10 ** -1,
    ]
    extend = "both"

    fig, _, cb2 = disc_cube_plot(
        dummy_lat_lon_cube(frac_diffs),
        bin_edges=diff_boundaries,
        ax=axes[2],
        cbar_label="<Ob. - Pr.)> / <Ob.>",
        extend=extend,
        cmap="PuOr_r",
        # Add labelled rectangles only to the last plot.
        aux0_label=aux0_label,
        aux1_label=aux1_label,
        loc=(0.8, 0.11),
        height=0.02,
        aspect=2.5,
        cmap_midpoint=0,
        cmap_symmetric=False,
        **plot_kwargs,
    )

    if cbar_label_x_offset is not None:
        # Manual control.
        max_x = 0
        for cb in (cb0, cb1, cb2):
            bbox = cb.ax.get_position()
            if bbox.xmax > max_x:
                max_x = bbox.xmax

        for cb in (cb0, cb1, cb2):
            bbox = cb.ax.get_position()
            mean_y = (bbox.ymin + bbox.ymax) / 2.0
            cb.ax.yaxis.set_label_coords(
                max_x + cbar_label_x_offset, mean_y, transform=fig.transFigure
            )
    else:
        fig.align_labels()

    for ax, title in zip(axes, ascii_lowercase):
        ax.text(*text_xy, f"({title})", transform=ax.transAxes)

    figure_saver.save_figure(
        fig,
        "ba_prediction" if filename is None else filename,
        sub_directory="predictions",
    )

    # Visualising the diff distribution.

    # diffs = frac_diffs.data[~frac_diffs.mask].ravel()
    # diffs = diffs[np.abs(diffs) > 1e-6]
    # signs = np.sign(diffs)
    # diffs = np.log10(np.abs(diffs))

    # plt.figure(dpi=300)
    # plt.title('1')
    # plt.hist(diffs[signs>0], bins=200)
    # plt.yscale('log')

    # plt.figure(dpi=300)
    # plt.title('-1')
    # plt.hist(diffs[signs<0], bins=200)
    # plt.yscale('log')


class SetupFourMapAxes:
    """Context manager than handles construction and formatting of map axes.

    A single shared colorbar axis is created.

    Examples:
        >>> with SetupFourMapAxes() as (fig, axes, cax):  # doctest: +SKIP
        >>>     # Carry out plotting here.
        >>> # Keep using `fig`, etc... here to carry out saving, etc...
        >>> # It is important this is done after __exit__ is called!

    """

    def __init__(self, figsize=(9.86, 4.93), cbar="vertical"):
        """Define basic parameters used to set up the figure and axes."""
        self.fig = plt.figure(figsize=figsize)
        self.cbar = cbar

        if self.cbar == "vertical":
            # Axis factor.
            af = 3
            nrows = 2 * af

            gs = self.fig.add_gridspec(
                nrows=nrows,
                ncols=2 * af + 2,
                width_ratios=[1 / af, 1 / af] * af + [0.001] + [0.02],
            )
            self.axes = [
                self.fig.add_subplot(
                    gs[i * af : (i + 1) * af, j * af : (j + 1) * af],
                    projection=ccrs.Robinson(),
                )
                for j, i in product(range(2), repeat=2)
            ]

            diff = 2
            assert (
                diff % 2 == 0
            ), f"Want an even diff for symmetric bar placement (got diff {diff})."
            cax_l = 0 + diff // 2
            cax_u = nrows - diff // 2

            self.cax = self.fig.add_subplot(gs[cax_l:cax_u, -1])
        elif self.cbar == "horizontal":
            # Axis factor.
            af = 3
            ncols = 2 * af

            gs = self.fig.add_gridspec(
                nrows=2 * af + 1,
                ncols=ncols,
                height_ratios=[1 / af, 1 / af] * af + [0.05],
            )
            self.axes = [
                self.fig.add_subplot(
                    gs[i * af : (i + 1) * af, j * af : (j + 1) * af],
                    projection=ccrs.Robinson(),
                )
                for j, i in product(range(2), repeat=2)
            ]

            diff = 4
            assert (
                diff % 2 == 0
            ), f"Want an even diff for symmetric bar placement (got diff {diff})."
            cax_l = 0 + diff // 2
            cax_u = ncols - diff // 2

            self.cax = self.fig.add_subplot(gs[-1, cax_l:cax_u])
        else:
            raise ValueError(f"Unkown value for 'cbar' {cbar}.")

    def __enter__(self):
        """Return the figure, 4 main plotting axes, and colorbar axes."""
        return self.fig, self.axes, self.cax

    def __exit__(self, exc_type, value, traceback):
        """Adjust axis positions after plotting."""
        if self.cbar == "vertical":
            self.fig.subplots_adjust(wspace=0, hspace=0.5)

            # Move the left-column axes to the right to decrease the gap.
            for ax in self.axes[0:2]:
                box = ax.get_position()
                shift = 0.015
                box.x0 += shift
                box.x1 += shift
                ax.set_position(box)
        elif self.cbar == "horizontal":
            self.fig.subplots_adjust(wspace=-0.43, hspace=0.5)

            self.cax.xaxis.set_label_position("top")

            # Move the legend axes upwards.
            ax = self.cax
            box = ax.get_position()
            shift = 0.01
            box.y0 += shift
            box.y1 += shift
            ax.set_position(box)


@cache
def get_aux0_aux1_kwargs(y_test, master_mask):
    """Calculate regions denoting available BA data.

    These regions are characterised by missing data from other datasets.

    """
    single_master_mask = master_mask[0]
    ba_data = ba_dataset_map[y_test.name]().get_mean_dataset().cube.data

    land_mask = get_land_mask()

    # Indicate areas with 0 BA but with BA data availability (but without data
    # availability otherwise).
    unique_ba_values = np.unique(ba_data)
    zero_ba = (ba_data.data < unique_ba_values[1]) & land_mask & single_master_mask

    # Indicate areas with nonzero BA but with BA data availability (but without data
    # availability otherwise).
    nonzero_ba = (
        (ba_data.data.data > unique_ba_values[0]) & land_mask & single_master_mask
    )
    return dict(
        aux0=zero_ba,
        aux0_label="BA = 0",
        aux1=nonzero_ba,
        aux1_label="BA > 0",
    )
