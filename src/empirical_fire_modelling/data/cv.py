# -*- coding: utf-8 -*-
import logging
import math
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from wildfires.analysis import cube_plotting

from ..cache import cache, mark_dependency
from .core import get_map_data

__all__ = (
    "buffered_leave_one_out",
    "generate_structure",
    "random_binary_dilation_split",
)

logger = logging.getLogger(__name__)


def generate_structure(N=7, size=1, verbose=False):
    """Generate a rank 2 structure.

    Args:
        N (int): Number of rows, columns of the structure.
        size (int): Determines the number of True elements. `0 <= size < N`.
        verbose (bool): Visualise the created structure.

    Returns:
        (N, N) numpy.ndarray: The generated structure.

    """
    diffs = np.abs(np.arange(N) - N // 2)
    structure = (diffs[np.newaxis] + diffs[:, np.newaxis]) <= size

    if verbose:
        plt.figure()
        plt.imshow(structure, cmap="Greys", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"N={N}, Size={size}, Total={np.sum(structure)}")

    return ((N, size, np.sum(structure)), structure)


@mark_dependency
def apply_structure(array, structure):
    """Apply a structure to an array akin to binary dilation.

    The structure is rolled along axis=1 (longitude) if needed, but clipped along
    axis=0 (latitude).

    Args:
        array ((M, N) array): Boolean array.
        structure ((S, S) array): Boolean array.

    Returns:
        (M, N) array: Boolean array.

    Raises:
        ValueError: If any of the arguments is not a 2D array.
        ValueError: If the number of true elements in `array` is not 1.
        ValueError: If `structure` is not square.
        ValueError: If `S` is not odd.
        ValueError: If there are False elements along the outside of `structure`.
        ValueError: If `S` exceeds `M` or `N`.

    """
    if array.ndim != 2 or structure.ndim != 2:
        raise ValueError("Both arrays need to be 2-dimensional.")
    if np.sum(array) != 1:
        raise ValueError("The number of True elements in array should be 1.")
    if structure.shape[0] != structure.shape[1]:
        raise ValueError("The structure array should be square.")
    if structure.shape[0] % 2 == 0:
        raise ValueError("The number of rows in structure should be odd.")
    if (
        not np.any(structure[:, 0])
        or not np.any(structure[:, -1])
        or not np.any(structure[0])
        or not np.any(structure[-1])
    ):
        raise ValueError(
            "The structure should not contain only False elements along its perimeter."
        )
    if np.any(np.array(structure.shape) > np.array(array.shape)):
        raise ValueError(
            "Size of the structure should not exceed the size of the array."
        )

    n = structure.shape[0]
    halve_n = n // 2
    central_indices = tuple(map(itemgetter(0), np.where(array)))

    bottom_clip = max(halve_n - central_indices[0], 0)
    top_clip = max(central_indices[0] + halve_n + 1 - array.shape[0], 0)
    structure = structure[bottom_clip : n - top_clip]

    middle_array_index = array.shape[1] // 2

    # Embed the structure in an empty array of the same size of `array`, then shift
    # the entries accordingly.
    canvas = np.zeros_like(array)
    canvas[
        central_indices[0]
        - halve_n
        + bottom_clip : central_indices[0]
        + halve_n
        + 1
        + top_clip,
        middle_array_index - halve_n : middle_array_index + halve_n + 1,
    ] = structure

    # Roll along the axis=1 (longitude).
    roll_amount = central_indices[1] - middle_array_index
    canvas = np.roll(canvas, roll_amount, axis=1)
    # assert False
    return canvas


@cache(dependencies=(get_map_data,), ignore=("verbose", "dpi"))
def random_binary_dilation_split(
    exog_data,
    endog_data,
    structure,
    master_mask,
    test_frac=0.05,
    train_frac=None,
    seed=0,
    verbose=False,
    dpi=400,
):
    """Split data with the test data surrounded by ignored data.

    The shape and quantity of ignored data is dictated by `structure`.

    Note: Need ~dpi=1400 to see the divisions clearly (`verbose=True`).

    Args:
        exog_data, endog_data (pd.DataFrame, pd.Series): Predictor and target
            variables.
        structure ((N, N) numpy.ndarray): Structure used to dictate ignored data
            around the test samples.
        master_mask (numpy.ndarray): Mask controlling mapping from `exog_data`,
            `endog_data` to mapped data.
        test_frac (float): Fraction of samples to reserve for testing.
        train_frac (float or None): Fraction of samples to use for training. If `None`
            is given, all possible samples will be used.
        seed (int): Random number generator seed used to dictate where test samples
            are located.
        verbose (bool): Plot the training and test masks.
        dpi (int): Figure dpi. Only used if `verbose` is True.

    Returns:
        desc_str: Descriptive string.
        (total_samples, n_ignored, n_train, n_hold_out): Split statistics.
        train_X: Training predictor data.
        hold_out_X: Test predictor data.
        train_y: Training target data.
        hold_out_y: Test training data.

    Raises:
        ValueError: If `master_mask` is not a rank 3 array.
        ValueError: If `master_mask` is not identical across each slice along its
            first (temporal) dimension.
        ValueError: If `train_frac` cannot be satisfied, e.g. because too many samples
            are being used for testing and exclusion zones around test samples.

    """
    if master_mask.ndim != 3:
        raise ValueError(f"Expected a rank 3 array, got: {master_mask.ndim}.")

    if not np.all(np.all(master_mask[:1] == master_mask, axis=0)):
        raise ValueError("'master_mask' was not identical across each temporal slice.")

    rng = np.random.default_rng(seed)
    collapsed_master_mask = master_mask[0]
    single_total_samples = np.sum(~collapsed_master_mask)
    total_samples = single_total_samples * master_mask.shape[0]

    possible_indices = np.array(list(zip(*np.where(~collapsed_master_mask))))

    # Per time slice.
    n_test_samples = round(single_total_samples * test_frac)

    # Select test data.
    test_indices = possible_indices[
        rng.choice(np.arange(len(possible_indices)), size=n_test_samples, replace=False)
    ]

    hold_out_selection = np.zeros_like(collapsed_master_mask)
    hold_out_selection[(test_indices[:, 0], test_indices[:, 1])] = True

    # Select data around the test data to ignore.
    ignored_data = ndimage.binary_dilation(hold_out_selection, structure) & (
        ~hold_out_selection
    )

    # The remaining data is then used for training, depending on train_frac.
    possible_train_selection = (
        ~(hold_out_selection | ignored_data) & ~collapsed_master_mask
    )

    if train_frac is None:
        train_selection = possible_train_selection
    else:
        possible_train_indices = np.array(
            list(zip(*np.where(possible_train_selection)))
        )
        n_train_samples = round(single_total_samples * train_frac)

        if len(possible_train_indices) < n_train_samples:
            raise ValueError(
                f"Need at least {n_train_samples} samples to satisfy train_frac: "
                f"{train_frac}, but only have {len(possible_train_indices)} "
                f"({len(possible_train_indices) / single_total_samples:0.4f})."
            )

        # Select train data.
        train_indices = possible_train_indices[
            rng.choice(
                np.arange(len(possible_train_indices)),
                size=n_train_samples,
                replace=False,
            )
        ]

        train_selection = np.zeros_like(collapsed_master_mask)
        train_selection[(train_indices[:, 0], train_indices[:, 1])] = True

    # Apply the master_mask to the training and test data to arrive at the final 3D mask.
    train_selection = train_selection[None] & (~master_mask)
    hold_out_selection = hold_out_selection[None] & (~master_mask)

    if verbose:
        # Plot a map of the selections.
        mask_vis = np.zeros_like(master_mask, dtype=np.int32)
        mask_vis[hold_out_selection] = 1
        mask_vis[train_selection] = 2
        cube_plotting(
            np.mean(mask_vis, axis=0),
            title=str(seed),
            fig=plt.figure(dpi=dpi),
        )

    # Transform X, y to 3D arrays before selecting using the above masks.
    mm_endog = get_map_data(endog_data.values, master_mask)
    train_y = mm_endog.data[train_selection]
    hold_out_y = mm_endog.data[hold_out_selection]

    # Repeat for all columns in X.
    train_X_data = {}
    hold_out_X_data = {}
    for col in exog_data.columns:
        mm_x_col = get_map_data(exog_data[col].values, master_mask)
        train_X_data[col] = mm_x_col.data[train_selection]
        hold_out_X_data[col] = mm_x_col.data[hold_out_selection]

    train_X = pd.DataFrame(train_X_data)
    hold_out_X = pd.DataFrame(hold_out_X_data)

    n_ignored = np.sum(ignored_data[None] & (~master_mask))
    n_train = np.sum(train_selection)
    n_hold_out = np.sum(hold_out_selection)

    desc_str = (
        f"Total samples: {total_samples:0.1e}, "
        f"Ignored: {100 * n_ignored / total_samples: 0.1f}%, "
        f"Train: {100 * n_train / total_samples: 0.1f}%, "
        f"Test: {100 * n_hold_out / total_samples: 0.1f}%"
    )

    return (
        desc_str,
        (total_samples, n_ignored, n_train, n_hold_out),
        train_X,
        hold_out_X,
        train_y,
        hold_out_y,
    )


@cache(dependencies=(get_map_data, apply_structure), ignore=("verbose", "dpi"))
def buffered_leave_one_out(
    exog_data,
    endog_data,
    master_mask,
    radius,
    max_rad,
    seed=0,
    max_tries=10,
    verbose=False,
    dpi=400,
):
    """Split data with a single test sample surrounded by ignored data.

    Data is excluded using a given number of radii up to a given maximum radius (in
    units of pixels, i.e. grid cells).

    Args:
        exog_data, endog_data (pd.DataFrame, pd.Series): Predictor and target
            variables.
        master_mask (numpy.ndarray): Mask controlling mapping from `exog_data`,
            `endog_data` to mapped data.
        radius (float): Radius (number of pixels) to exclude.
        max_rad (float): Maximum radius to be attempted.
        seed (int): Random number generator seed used to dictate where test samples
            are located.
        max_tries (int): Number of allowed attempts to find a test sample that is
            within the train observations, given the maximum excluded radius `max_rad`.
        verbose (bool): Plot the training and test masks.
        dpi (int): Figure dpi. Only used if `verbose` is True.

    Returns:
        n_ignored (int): Number of ignored samples.
        n_train (int): Number of train samples.
        n_test (int): Number of test samples.
        total_samples (int): Total number of samples.
        train_X: Training predictor data.
        test_X: Test predictor data.
        train_y: Training target data.
        test_y: Test training data.

    Raises:
        ValueError: If `master_mask` is not a rank 3 array.
        ValueError: If `master_mask` is not identical across each slice along its
            first (temporal) dimension.
        ValueError: If `radius` is larger than `max_rad`.
        RuntimeError: If no suitable test site can be found within `max_tries`.

    """
    if master_mask.ndim != 3:
        raise ValueError(f"Expected a rank 3 array, got: {master_mask.ndim}.")

    if not np.all(np.all(master_mask[:1] == master_mask, axis=0)):
        raise ValueError("'master_mask' was not identical across each temporal slice.")

    if radius > (max_rad + 1e-7):
        raise ValueError("'radius' was larger than 'max_rad'")

    rng = np.random.default_rng(seed)

    collapsed_master_mask = master_mask[0]
    single_total_samples = np.sum(~collapsed_master_mask)
    total_samples = single_total_samples * master_mask.shape[0]

    possible_indices = np.array(list(zip(*np.where(~collapsed_master_mask))))

    def get_structure(radius):
        # Generate a rank 2 structure.
        N = math.ceil(radius * 2)
        if N % 2 == 0:
            # Ensure there is an odd number of elements. This results in a symmetric
            # structure.
            N += 1

        if N > 1:
            # Calculate the differences to the central index.
            diffs = (np.arange(N) - N // 2) ** 2
            structure = np.sqrt(diffs[np.newaxis] + diffs[:, np.newaxis]) <= (
                radius + 1e-7
            )
        else:
            structure = np.array([[True]])

        if verbose:
            plt.figure()
            plt.imshow(structure, cmap="Greys", vmin=0, vmax=1)
            plt.axis("off")
            plt.title(f"N={N}, Total={np.sum(structure)}")

        return structure

    structure = get_structure(radius)
    max_rad_structure = get_structure(max_rad)

    n_train_samples = single_total_samples - np.sum(max_rad_structure)

    tries = 0

    while tries < max_tries:
        # Select a single test sample.
        test_indices = possible_indices[
            rng.integers(low=0, high=len(possible_indices), size=(1,))
        ]
        hold_out_selection = np.zeros_like(collapsed_master_mask)
        hold_out_selection[(test_indices[:, 0], test_indices[:, 1])] = True

        # Select data around the test sample to ignore.
        ignored_data = apply_structure(hold_out_selection, structure) & (
            ~hold_out_selection
        )
        max_rad_ignored_data = apply_structure(
            hold_out_selection, max_rad_structure
        ) & (~hold_out_selection)

        # The remaining data is then used for training, depending on train_frac.
        possible_train_selection = (
            ~(hold_out_selection | ignored_data) & ~collapsed_master_mask
        )
        max_rad_possible_train_selection = (
            ~(hold_out_selection | max_rad_ignored_data) & ~collapsed_master_mask
        )

        possible_train_indices = np.array(
            list(zip(*np.where(possible_train_selection)))
        )

        if len(possible_train_indices) < n_train_samples:
            raise ValueError(
                f"Need at least {n_train_samples} samples, but only have "
                f"{len(possible_train_indices)}."
            )

        # Select train data.
        train_indices = possible_train_indices[
            rng.choice(
                np.arange(len(possible_train_indices)),
                size=n_train_samples,
                replace=False,
            )
        ]

        train_selection = np.zeros_like(collapsed_master_mask)
        train_selection[(train_indices[:, 0], train_indices[:, 1])] = True

        max_rad_train_selection = train_selection & max_rad_possible_train_selection

        # Apply the master_mask to the training and test data to arrive at the final 3D mask.
        train_selection = train_selection[None] & (~master_mask)
        hold_out_selection = hold_out_selection[None] & (~master_mask)

        max_rad_train_selection = max_rad_train_selection[None] & (~master_mask)

        if verbose:
            # Plot a map of the selections.
            mask_vis = np.zeros_like(master_mask, dtype=np.int32)
            mask_vis[hold_out_selection] = 1
            mask_vis[train_selection] = 2
            cube_plotting(
                np.mean(mask_vis, axis=0),
                title=str(seed),
                fig=plt.figure(dpi=dpi),
            )

        # Transform X, y to 3D arrays before selecting using the above masks.
        mm_endog = get_map_data(endog_data.values, master_mask)
        train_y = mm_endog.data[train_selection]
        hold_out_y = mm_endog.data[hold_out_selection]

        # Repeat for all columns in X.
        train_X_data = {}
        hold_out_X_data = {}

        max_rad_train_X_data = {}

        for col in exog_data.columns:
            mm_x_col = get_map_data(exog_data[col].values, master_mask)
            train_X_data[col] = mm_x_col.data[train_selection]
            hold_out_X_data[col] = mm_x_col.data[hold_out_selection]

            max_rad_train_X_data[col] = mm_x_col.data[max_rad_train_selection]

        train_X = pd.DataFrame(train_X_data)
        hold_out_X = pd.DataFrame(hold_out_X_data)

        max_rad_train_X = pd.DataFrame(max_rad_train_X_data)

        # Verify that test data for the largest radius is within the range of
        # observations in the train data.
        if np.all(
            np.max(hold_out_X.values, axis=0) <= np.max(max_rad_train_X.values, axis=0)
        ) and np.all(
            np.min(hold_out_X.values, axis=0) >= np.min(max_rad_train_X.values, axis=0)
        ):
            n_ignored = np.sum(ignored_data[None] & (~master_mask))
            n_train = np.sum(train_selection)
            n_hold_out = np.sum(hold_out_selection)

            assert np.sum(max_rad_train_selection) <= n_train

            return (
                n_ignored,
                n_train,
                n_hold_out,
                total_samples,
                train_X,
                hold_out_X,
                train_y,
                hold_out_y,
            )
        logger.warning("Trying another sample location.")
        tries += 1

    raise RuntimeError("No suitable site could be found.")
