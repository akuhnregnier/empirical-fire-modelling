# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

from ..utils import transform_series_sum_norm

__all__ = ("plot_and_list_importances",)


def plot_and_list_importances(importances, methods, print_n=15, N=15, verbose=True):
    fig, ax = plt.subplots()

    transformed = {}

    combined = None
    for method in methods:
        transformed[method] = transform_series_sum_norm(importances[method])
        if combined is None:
            combined = transformed[method].copy()
        else:
            combined += transformed[method]
    combined.sort_values(ascending=False, inplace=True)

    transformed = pd.DataFrame(transformed).reindex(combined.index, axis=0)

    for method, marker in zip(methods, ["o", "x", "s", "^"]):
        ax.plot(
            transformed[method], linestyle="", marker=marker, label=method, alpha=0.8
        )
    ax.set_xticklabels(
        transformed.index, rotation=45 if len(transformed.index) <= 15 else 90
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.4)
    if verbose:
        print(combined[:print_n].to_latex())
    return combined[:N]
