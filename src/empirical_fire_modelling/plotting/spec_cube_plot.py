# -*- coding: utf-8 -*-
import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import from_levels_and_colors

from .utils import get_sci_format


def disc_cube_plot(
    cube,
    bin_edges,
    fig=None,
    ax=None,
    cax=None,
    cmap="inferno_r",
    aux0=None,
    aux1=None,
    extend="neither",
    aux0_label="",
    aux1_label="",
    aux0_c=np.array([150, 150, 150, 200], dtype=np.float64) / 255,
    aux1_c=np.array([64, 64, 64, 200], dtype=np.float64) / 255,
    cbar_label="",
    cbar_orientation="vertical",
    cbar_fraction=0.02,
    cbar_pad=0.07,
    cbar_aspect=24,
    cbar_shrink=0.6,
    cbar_extendfrac=0.07,
    cbar_anchor=(0.5, 1.0),
    cbar_panchor=(0.5, 0.0),
    cbar_format=get_sci_format(ndigits=1),
    loc=(0.7, 0.2),
    height=0.04,
    aspect=2,
    spacing=0.04 * 0.2,
    cmap_midpoint=None,
    cmap_symmetric=False,
):
    """Plotting of cube using given bin edges."""
    if not isinstance(cube, iris.cube.Cube):
        raise ValueError("cube is not an iris Cube.")
    if len(cube.shape) != 2:
        raise ValueError("cube is not 2-dimensional.")

    if cmap_symmetric and cmap_midpoint is None:
        raise ValueError("If cmap_symmetric is True, cmap_midpoint has to be given.")
    if cmap_midpoint is not None and np.sum(np.isclose(bin_edges, cmap_midpoint)) != 1:
        raise ValueError("cmap_midpoint has to match one of the given bin edges.")

    gridlons = cube.coord("longitude").contiguous_bounds()
    gridlats = cube.coord("latitude").contiguous_bounds()

    data = cube.data

    if not isinstance(data, np.ma.MaskedArray):
        raise ValueError("Data should be a MaskedArray.")

    projection = ccrs.Robinson()

    if fig is None and ax is None:
        fig = plt.figure()
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes(projection=projection)

    cmap_slice = slice(None)
    try:
        orig_cmap = plt.get_cmap(cmap)
    except ValueError:
        if isinstance(cmap, str) and "_r" in cmap:
            # Try to reverse the colormap manually, in case a reversed colormap was
            # requested using the '_r' suffix, but this is not available.
            cmap = cmap.rstrip("_r")
            orig_cmap = plt.get_cmap(cmap)

            # Flip limits to achieve reversal effect.
            cmap_slice = slice(None, None, -1)

    cmap_sample_lims = [0, 1]

    if extend == "neither":
        add_colors = -1
        assert np.min(data) >= bin_edges[0]
        assert np.max(data) <= bin_edges[-1]
    elif extend in ("min", "max"):
        add_colors = 0
        if extend == "min":
            assert np.max(data) <= bin_edges[-1]
        else:
            assert np.min(data) >= bin_edges[0]
    elif extend == "both":
        add_colors = 1

    n_colors = len(bin_edges) + add_colors

    if cmap_midpoint is None:
        colors = orig_cmap(np.linspace(*cmap_sample_lims[cmap_slice], n_colors))
    else:
        if cmap_symmetric:
            # Adjust the colormap sample limits such that the deviation from
            # 0.5 is proportional to the magnitude of the maximum deviation
            # from the midpoint.
            diffs = np.array(
                (bin_edges[0] - cmap_midpoint, bin_edges[-1] - cmap_midpoint)
            )
            max_diff = max(np.abs(diffs))
            scaled = diffs / max_diff
            cmap_sample_lims = 0.5 + scaled * 0.5
        # Find closest bin edge.
        closest_bound_index = np.argmin(
            np.abs(np.asarray(bin_edges[cmap_slice]) - cmap_midpoint)
        )

        lower_range = 0.5 - cmap_sample_lims[0]
        n_lower = closest_bound_index + (1 if extend in ("min", "both") else 0)

        upper_range = cmap_sample_lims[1] - 0.5
        n_upper = (
            len(bin_edges)
            - 1
            - closest_bound_index
            + (1 if extend in ("max", "both") else 0)
        )

        colors = np.vstack(
            (
                orig_cmap(
                    cmap_sample_lims[0]
                    + np.arange(n_lower) * (2 * lower_range / (1 + 2 * n_lower))
                ),
                orig_cmap(
                    cmap_sample_lims[1]
                    - np.arange(n_upper - 1, -1, -1)
                    * (2 * upper_range / (1 + 2 * n_upper))
                ),
            )
        )[cmap_slice]

    cmap, norm = from_levels_and_colors(
        levels=list(bin_edges),
        colors=colors,
        # 'neither', 'min', 'max', 'both'
        extend=extend,
    )

    mesh = ax.pcolormesh(
        gridlons,
        gridlats,
        data,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        transform=ccrs.PlateCarree(),
    )

    cbar = fig.colorbar(
        mesh,
        label=cbar_label,
        orientation=cbar_orientation,
        fraction=cbar_fraction,
        pad=cbar_pad,
        aspect=cbar_aspect,
        shrink=cbar_shrink,
        extendfrac=cbar_extendfrac,
        anchor=cbar_anchor,
        panchor=cbar_panchor,
        format=cbar_format,
        cax=cax,
        ax=ax,
    )

    if aux0 is not None or aux1 is not None:
        if aux0 is not None and aux1 is not None:
            # Ensure they do not overlap.
            assert not np.any(aux0 & aux1)

        # Draw data including these auxiliary points.
        data = data.copy()

        min_data = np.min(data)

        if aux0 is not None:
            data[aux0 & data.mask] = min_data - 2
        if aux1 is not None:
            data[aux1 & data.mask] = min_data - 1

        colors = np.vstack((np.array([aux0_c, aux1_c]), colors))

        cmap, norm = from_levels_and_colors(
            levels=[
                min_data - 2.5,
                min_data - 1.5,
                *((min_data - 0.5,) if extend in ("min", "both") else ()),
            ]
            + list(bin_edges),
            colors=colors,
            extend="max" if extend in ("max", "both") else "neither",
        )

        mesh = ax.pcolormesh(
            gridlons,
            gridlats,
            data,
            cmap=cmap,
            norm=norm,
            rasterized=True,
            transform=ccrs.PlateCarree(),
        )

        # Do not re-draw colorbar including the aux0 and aux1 colours.
        # Instead, add rectangles to indicate the meaning of the auxiliary levels.

        width = height * aspect

        label_kwargs = dict(
            x=loc[0] + width + spacing,
            transform=fig.transFigure,
            verticalalignment="center",
        )

        rect_kwargs = dict(
            width=width,
            height=height,
            fill=True,
            alpha=1,
            zorder=1000,
            transform=fig.transFigure,
            figure=fig,
        )

        if aux0 is not None and aux0_label:
            fig.patches.extend(
                [
                    plt.Rectangle(
                        loc,
                        color=aux0_c,
                        **rect_kwargs,
                    ),
                ]
            )
            ax.text(y=loc[1] + height / 2, s=aux0_label, **label_kwargs)
        if aux1 is not None and aux1_label:
            fig.patches.extend(
                [
                    plt.Rectangle(
                        (loc[0], loc[1] - height - spacing), color=aux1_c, **rect_kwargs
                    ),
                ]
            )
            ax.text(y=loc[1] - height / 2 - spacing, s=aux1_label, **label_kwargs)

    ax.gridlines(zorder=0, alpha=0.4, linestyle="--", linewidth=0.3)
    ax.coastlines(resolution="110m", linewidth=0.5)

    return fig, ax, cbar
