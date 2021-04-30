# -*- coding: utf-8 -*-
import math
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ..exceptions import EmptyUnitSpecError


def get_sci_format(ndigits=1, zero_thres=1e-15, zero_str="0"):
    @FuncFormatter
    def fmt(x, pos):
        if abs(x) < zero_thres:
            return zero_str
        exponent = math.floor(math.log10(abs(x)))
        mantissa = format(round(x / (10 ** exponent), ndigits=ndigits), f".{ndigits}f")
        return rf"${mantissa} \times 10^{{{exponent}}}$"

    return fmt


def _update_label(old, exponent_text):
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
        ax.set_label_text(_update_label(label, exponent_text))
