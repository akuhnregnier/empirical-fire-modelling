# -*- coding: utf-8 -*-
import math
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..exceptions import EmptyUnitSpecError


def get_sci_format(
    ndigits=1,
    float_thres=1e-15,
    zero_str="0",
    atol=1e-8,
    atol_exceeded="raise",
    trim_leading_one=False,
):
    """Scientific formatter.

    Args:
        ndigits (int): Number of digits.
        float_thres (float): Threshold for float comparisons.
        zero_str (str): If the number if within `float_thres` of 0, use this instead.
        atol (float): Absolute tolerance to detect incorrect labels.
        atol_exceeded ({'raise', 'adjust'}): If 'raise', raise ValueError if the
            tolernace is exceeded by a formatted label. If 'adjust', use `ndigits=10`
            for this value instead.
        trim_leading_one (bool): If the formatted label starts with '1 x', trim
            '1 x' off the beginning.

    """

    @FuncFormatter
    def fmt(x, pos):
        if abs(x) < float_thres:
            return zero_str
        exponent = math.floor(math.log10(abs(x)))
        rounded = round(x / (10 ** exponent), ndigits=ndigits)
        new = rounded * 10 ** exponent
        if abs(new - x) > atol:
            if atol_exceeded == "adjust":
                # Use a higher number of ndigits.
                exponent = math.floor(math.log10(abs(x)))
                rounded = round(x / (10 ** exponent), ndigits=10)
                new = rounded * 10 ** exponent
                mantissa = format(rounded, f".{10}f")
            else:
                raise ValueError(
                    f"Discrepancy too large - after rounding: {new} vs. original: {x}"
                )
        else:
            mantissa = format(rounded, f".{ndigits}f")

        if trim_leading_one and abs(float(mantissa) - 1) < float_thres:
            # Trim the leading '1 x'.
            return rf"$10^{{{exponent}}}$"
        return rf"${mantissa} \times 10^{{{exponent}}}$"

    return fmt


def get_float_format(
    factor=1,
    ndigits=1,
    float_thres=1e-15,
    zero_str=None,
    atol=1e-8,
    atol_exceeded="raise",
):
    """Scientific formatter.

    Args:
        factor (float): Factor to divide each displayed label by.
        ndigits (int): Number of digits.
        float_thres (float): Threshold for 0.
        zero_str (str or None): If the number if within `float_thres` of 0, use this
            instead, except for if None is given.
        atol (float): Absolute tolerance to detect incorrect labels.
        atol_exceeded ({'raise', 'adjust'}): If 'raise', raise ValueError if the
            tolernace is exceeded by a formatted label. If 'adjust', use `ndigits=10`
            for this value instead.

    """

    @FuncFormatter
    def fmt(x, pos):
        x /= factor
        if zero_str is not None and abs(x) < float_thres:
            return zero_str
        rounded = round(x, ndigits=ndigits)
        if abs(rounded - x) > atol:
            if atol_exceeded == "adjust":
                # Use a higher number of ndigits.
                rounded = round(x, ndigits=10)
            else:
                raise ValueError(
                    f"Discrepancy too large - after rounding: {rounded} vs. original: {x}"
                )
        return format(rounded, f"0.{ndigits}f")

    return fmt


def update_label_with_exp(old, exponent_text):
    """Update a label using a given exponent.

    Adapted from: http://greg-ashton.physics.monash.edu/setting-nice-axes-labels-in-matplotlib.html

    """
    if exponent_text == "":
        return old

    try:
        *_, last = re.finditer(r"\(.*?\)", old)
    except ValueError:
        units = ""
        label = old.strip()
    else:
        units = last.group(0)[1:-1]  # Trim parentheses.
        # Remove the units from the old label.
        label = old[: last.span()[0]].strip()
        if not units:
            raise EmptyUnitSpecError(f"Encountered empty unit spec () in label {old}.")

    exponent_text = exponent_text.replace("\\times", "")
    if units == "1":
        combined = exponent_text
    else:
        combined = f"{exponent_text} {units}"

    return f"{label} ({combined.strip()})"


def format_label_string_with_exponent(ax, axis="both"):
    """Format the label string with the exponent from the ScalarFormatter.

    Adapted from: http://greg-ashton.physics.monash.edu/setting-nice-axes-labels-in-matplotlib.html

    """
    ax.ticklabel_format(axis=axis, style="sci")

    axes_instances = []
    if axis in ("x", "both"):
        axes_instances.append(ax.xaxis)
    if axis in ("y", "both"):
        axes_instances.append(ax.yaxis)

    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw()  # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label_with_exp(label, exponent_text))
