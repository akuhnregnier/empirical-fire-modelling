# -*- coding: utf-8 -*-
import math

from matplotlib.ticker import FuncFormatter


def get_sci_format(ndigits=1, zero_thres=1e-15, zero_str="0"):
    @FuncFormatter
    def fmt(x, pos):
        if abs(x) < zero_thres:
            return zero_str
        exponent = math.floor(math.log10(abs(x)))
        mantissa = format(round(x / (10 ** exponent), ndigits=ndigits), f".{ndigits}f")
        return rf"${mantissa} \times 10^{{{exponent}}}$"

    return fmt
