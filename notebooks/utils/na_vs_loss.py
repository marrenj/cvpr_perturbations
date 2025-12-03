from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableSequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_neural_alignment_vs_loss(
    df: pd.DataFrame,
    *,
    roi_order: Optional[Iterable[str]] = None,
    plot_rois: Optional[Iterable[str]] = None,
    label_rois: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
    line_kwargs: Optional[Mapping[str, Mapping[str, float]]] = None,
    label_offsets: Optional[Mapping[str, Dict[str, float]]] = None,
    legend_title: str = "ROI",
    xtick_step: int = 20,
) -> plt.Axes:
    """
    Plot neural RSA alignment (rho) versus test loss for multiple ROIs.

    Parameters
    ----------
    df:
        DataFrame containing columns `roi`, `test_loss`, `rho`, and `epoch`.
    roi_order:
        Optional iterable describing the order in which ROIs are plotted. Defaults
        to the alphabetical order present in the DataFrame.
    plot_rois:
        Optional iterable specifying which ROIs to include in the figure. All ROIs
        are plotted when omitted.
    label_rois:
        Optional iterable of ROI names to label directly on the plot. Other ROIs
        will appear only in the legend.
    ax:
        Existing matplotlib axis; a new axis is created if omitted.
    line_kwargs:
        Mapping from ROI name to keyword arguments forwarded to `Axes.plot`.
        A fallback style (linewidth=0.8) is used when an ROI is missing.
    label_offsets:
        Mapping from ROI name to dicts with optional `dx`/`dy` adjustments for the
        annotation position.
    legend_title:
        Title shown above the ROI legend.
    xtick_step:
        Spacing between x-axis tick marks (after inversion).

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the rendered neural-alignment curve(s).
    """
    required_columns = {"roi", "test_loss", "rho"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    plot_rois_set = set(plot_rois) if plot_rois is not None else None
    plot_df = (
        df[df["roi"].isin(plot_rois_set)].copy()
        if plot_rois_set is not None
        else df.copy()
    )

    if plot_df.empty:
        raise ValueError("No rows available after applying the `plot_rois` filter.")

    ax = ax or plt.gca()
    available_rois = sorted(plot_df["roi"].unique())
    roi_order = list(roi_order) if roi_order is not None else available_rois
    if plot_rois_set is not None:
        roi_order = [roi for roi in roi_order if roi in plot_rois_set]
    roi_order = [roi for roi in roi_order if roi in available_rois]

    label_rois = set(label_rois) if label_rois is not None else set()
    if label_rois:
        label_rois &= set(available_rois)
    label_offsets = label_offsets or {}

    handles: MutableSequence[plt.Line2D] = []
    labels: MutableSequence[str] = []

    for roi in roi_order:
        roi_df = plot_df[plot_df["roi"] == roi]
        if roi_df.empty:
            continue

        defaults = {"linewidth": 0.8}
        roi_kwargs = {**defaults, **(line_kwargs.get(roi, {}) if line_kwargs else {})}
        (line,) = ax.plot(roi_df["test_loss"], roi_df["rho"], **roi_kwargs)

        if roi in label_rois:
            # Use the last point (closest to min test loss) for labeling.
            x = roi_df["test_loss"].iloc[-1]
            y = roi_df["rho"].iloc[-1]
            offsets = label_offsets.get(roi, {})
            dx = offsets.get("dx", -5)
            dy = offsets.get("dy", 0.0)

            ax.text(
                x + dx,
                y + dy,
                roi,
                color=line.get_color(),
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.7),
            )
        else:
            handles.append(line)
            labels.append(roi)

    if handles and labels:
        legend = ax.legend(
            handles,
            labels,
            title=legend_title,
            fontsize=7,
            title_fontsize=8,
        )
        for legline in legend.get_lines():
            legline.set_linewidth(1.5)

    ax.set_xlabel("Test Loss", fontweight="bold", fontsize=14)
    ax.set_ylabel("Neural Alignment", fontweight="bold", fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.invert_xaxis()

    xmin, xmax = ax.get_xlim()
    if xmin > xmax:
        start = int(np.ceil(xmin / xtick_step) * xtick_step)
        stop = int(np.floor(xmax / xtick_step) * xtick_step) - xtick_step
        xticks = np.arange(start, stop - xtick_step, -xtick_step)
    else:
        start = int(np.floor(xmin / xtick_step) * xtick_step)
        stop = int(np.ceil(xmax / xtick_step) * xtick_step) + xtick_step
        xticks = np.arange(start, stop + xtick_step, xtick_step)
    ax.set_xticks(xticks)

    return ax