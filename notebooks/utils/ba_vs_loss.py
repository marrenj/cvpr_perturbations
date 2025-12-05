from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_behavior_vs_loss(
    df: pd.DataFrame,
    *,
    label_epochs: Sequence[int] = (1, 5, 10, 15, 98),
    ax: plt.Axes | None = None,
    line_kwargs: Mapping[str, float] | None = None,
) -> plt.Axes:
    """
    Plot the behavioral-alignment-vs-test-loss S-curve used throughout the analysis notebook.

    Parameters
    ----------
    df:
        DataFrame containing at least `epoch`, `test_loss`, and `behavioral_rsa_rho`.
        The rows should appear in chronological order.
    label_epochs:
        Iterable of epoch numbers to annotate on the curve.
    ax:
        Existing matplotlib axis to draw on. A new axis is created when omitted.
    line_kwargs:
        Additional keyword arguments forwarded to `Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the rendered plot.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    
    required_columns = {"epoch", "test_loss", "behavioral_rsa_rho"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    ax = ax or plt.gca()
    line_kwargs = {"linewidth": 0.8, "color": "black", **(line_kwargs or {})}
    color = line_kwargs["color"]

    ax.plot(df["test_loss"], df["behavioral_rsa_rho"], **line_kwargs)

    for epoch in label_epochs:
        match = df[df["epoch"] == epoch]
        if match.empty:
            continue
        x = match["test_loss"].iloc[0]
        y = match["behavioral_rsa_rho"].iloc[0]

        if epoch == label_epochs[0]:
            label_x = x
            label_y = y + 0.05
            ha = "left"
        else:
            label_x = x + 7
            label_y = y + 0.02
            ha = "right"

        ax.text(label_x, label_y, f"epoch {epoch}", fontsize=7, ha=ha, va="bottom")
        ax.plot([x, label_x], [y, label_y], color=color, linewidth=0.7, zorder=2)

    ax.invert_xaxis()
    ax.set_xlabel("Test Loss", fontweight="bold", fontsize=14)
    ax.set_ylabel("Behavioral\nAlignment", fontweight="bold", fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    xmin, xmax = ax.get_xlim()
    step = 20
    if xmin > xmax:
        start = int(np.ceil(xmin / step) * step)
        stop = int(np.floor(xmax / step) * step) - step
        xticks = np.arange(start, stop - step, -step)
    else:
        start = int(np.floor(xmin / step) * step)
        stop = int(np.ceil(xmax / step) * step) + step
        xticks = np.arange(start, stop + step, step)
    ax.set_xticks(xticks)

    return ax


