# -*- coding: utf-8 -*-
from .configuration import (
    aux0_c,
    aux1_c,
    experiment_color_dict,
    experiment_colors,
    experiment_marker_dict,
    experiment_markers,
    experiment_plot_kwargs,
    experiment_zorder_dict,
    lag_color_dict,
    lag_colors,
    plotting_experiments,
)
from .core import *
from .spec_cube_plot import disc_cube_plot
from .utils import (
    format_label_string_with_exponent,
    get_float_format,
    get_sci_format,
    update_label_with_exp,
)
