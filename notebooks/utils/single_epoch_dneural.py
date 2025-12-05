from __future__ import annotations

import re
import math
from typing import Callable, Mapping, MutableMapping, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_single_epoch_delta_neural_alignment"]

EpochParser = Callable[[str], int]


def _infer_epoch_from_name(run_name: str) -> int:
    """Best-effort parser that extracts the trailing integer from a run name."""
    digits = re.findall(r"\d+", run_name)
    if not digits:
        raise ValueError(
            "Unable to infer perturbation epoch from run name "
            f"'{run_name}'. Provide `perturbation_epochs` or `epoch_parser`."
        )
    return int(digits[-1])


def plot_single_epoch_delta_neural_alignment(
    baseline_df: pd.DataFrame,
    perturbed_runs: Union[Mapping[str, pd.DataFrame], pd.DataFrame],
    *,
    roi_column: str = "roi",
    value_column: str = "rho",
    perturbed_runs_is_mapping: bool = True,
    training_run_column: str = "training_run",
    perturbation_epochs: Mapping[str, int] | None = None,
    epoch_parser: EpochParser | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    positive_color: str = "#2c7bb6",
    negative_color: str = "#d7191c",
    zero_color: str = "#7f8c8d",
    bar_kwargs: Mapping[str, float] | None = None,
    plot_rois: Sequence[str] | None = None,
) -> tuple[
    Union[plt.Axes, plt.Figure],
    Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
]:
    """
    Plot perturbation-induced delta in neural alignment for single-epoch sweeps.

    Parameters
    ----------
    baseline_df:
        DataFrame containing the unperturbed neural RSA metrics with columns
        `epoch`, `training_run`, `roi`, and the column indicated by `value_column`.
    perturbed_runs:
        Either a mapping from run identifier to DataFrame (each containing `epoch`,
        `training_run`, `roi`, and the neural-alignment column) or a single DataFrame
        that includes all perturbation runs with a `training_run` column.
    roi_column:
        Column containing ROI labels (defaults to `roi`).
    value_column:
        Column containing the neural alignment metric (defaults to `rho`).
    perturbed_runs_is_mapping:
        When True (default), `perturbed_runs` must be a mapping. When False, a single
        DataFrame is expected and runs are separated via `training_run_column`.
    training_run_column:
        Column name identifying runs inside `perturbed_runs` when
        `perturbed_runs_is_mapping=False`.
    perturbation_epochs:
        Optional mapping specifying the perturbation epoch for each run. When
        omitted, `epoch_parser` (or the built-in parser) is used to infer the
        epoch from the run name.
    epoch_parser:
        Callable that receives a run name and returns the perturbation epoch,
        used only when `perturbation_epochs` does not provide a value.
    ax:
        Existing matplotlib axis to draw on; a new axis is created when omitted.
    title:
        Optional plot title.
    positive_color / negative_color / zero_color:
        Bar colors used for positive, negative, and zero deltas respectively.
    bar_kwargs:
        Additional keyword arguments forwarded to `Axes.bar`.
    plot_rois:
        Optional iterable of ROI names to include when computing the delta. All ROIs
        are used when omitted.

    Returns
    -------
    tuple
        When a single ROI (or no ROI filter) is used, returns `(Axes, DataFrame)`
        describing the delta averaged across the selected ROIs. When
        `plot_rois` contains multiple ROIs, returns `(Figure, Dict[str, DataFrame])`
        where each subplot visualizes one ROI and the dictionary stores the
        underlying per-ROI delta values.
    """

    plt.rcParams["font.family"] = "Times New Roman"

    if plot_rois is not None:
        roi_order = list(dict.fromkeys(plot_rois))
        if not roi_order:
            raise ValueError("`plot_rois` must contain at least one ROI name.")
        roi_filter_set = set(roi_order)
    else:
        roi_order = None
        roi_filter_set = None
    multi_roi = roi_filter_set is not None and len(roi_filter_set) > 1

    required_cols = {"epoch", roi_column, value_column}
    missing = required_cols - set(baseline_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Baseline DataFrame missing required columns: {missing_str}"
        )

    if roi_filter_set is not None:
        baseline_df = baseline_df[baseline_df[roi_column].isin(roi_filter_set)].copy()
        if baseline_df.empty:
            raise ValueError(
                "No baseline rows remain after applying the specified `plot_rois`."
            )

    epoch_lookup: MutableMapping[str, int] = (
        {k: int(v) for k, v in (perturbation_epochs or {}).items()}
    )
    parser = epoch_parser or _infer_epoch_from_name

    records: list[dict[str, float | int | str]] = []
    roi_records: dict[str, list[dict[str, float | int | str]]] = {}

    if multi_roi:
        baseline_roi_lookup = baseline_df.set_index([roi_column, "epoch"])[value_column]
        baseline_epoch_mean = None
    else:
        baseline_epoch_mean = (
            baseline_df.groupby("epoch")[value_column].mean().rename("baseline_mean")
        )
        baseline_roi_lookup = None

    if perturbed_runs_is_mapping:
        if not isinstance(perturbed_runs, Mapping):
            raise TypeError(
                "Expected `perturbed_runs` to be a mapping when "
                "`perturbed_runs_is_mapping=True`."
            )
        run_items = list(perturbed_runs.items())
    else:
        if isinstance(perturbed_runs, Mapping):
            raise TypeError(
                "Set `perturbed_runs_is_mapping=True` when passing a mapping of runs."
            )
        if not isinstance(perturbed_runs, pd.DataFrame):
            raise TypeError(
                "Expected `perturbed_runs` to be a pandas DataFrame when "
                "`perturbed_runs_is_mapping=False`."
            )
        if training_run_column not in perturbed_runs.columns:
            raise ValueError(
                f"Column '{training_run_column}' is required in `perturbed_runs` "
                "when `perturbed_runs_is_mapping=False`."
            )
        run_items = [
            (
                f"training_run{int(run_id)}",
                run_df.reset_index(drop=True),
            )
            for run_id, run_df in perturbed_runs.groupby(training_run_column)
        ]

    for run_name, df in run_items:
        if df is None or df.empty:
            continue

        missing = required_cols - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Run '{run_name}' is missing required columns: {missing_str}"
            )

        if run_name in epoch_lookup:
            perturb_epoch = epoch_lookup[run_name]
        else:
            try:
                perturb_epoch = parser(run_name)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Failed to infer perturbation epoch for run "
                    f"'{run_name}'. Supply `perturbation_epochs` explicitly."
                ) from exc

        run_at_epoch = df[df["epoch"] == perturb_epoch]
        if roi_filter_set is not None:
            run_at_epoch = run_at_epoch[run_at_epoch[roi_column].isin(roi_filter_set)]
        if run_at_epoch.empty:
            continue

        if multi_roi:
            grouped = run_at_epoch.groupby(roi_column)
            for roi, roi_rows in grouped:
                if roi_filter_set is not None and roi not in roi_filter_set:
                    continue
                roi_value = float(roi_rows[value_column].iloc[0])
                try:
                    baseline_value = float(
                        baseline_roi_lookup.loc[(roi, perturb_epoch)]
                    )
                except KeyError:
                    continue

                roi_records.setdefault(roi, []).append(
                    {
                        "run_name": run_name,
                        "perturbation_epoch": perturb_epoch,
                        "delta_neural_alignment": roi_value - baseline_value,
                        "perturbed_neural_alignment": roi_value,
                        "baseline_neural_alignment": baseline_value,
                    }
                )
        else:
            run_mean = run_at_epoch[value_column].mean()
            try:
                baseline_mean = baseline_epoch_mean.loc[perturb_epoch]
            except KeyError:
                continue

            delta = run_mean - float(baseline_mean)

            records.append(
                {
                    "run_name": run_name,
                    "perturbation_epoch": perturb_epoch,
                    "delta_neural_alignment": delta,
                    "perturbed_neural_alignment_mean": run_mean,
                    "baseline_neural_alignment_mean": float(baseline_mean),
                }
            )

    bar_defaults = {"alpha": 0.7, "edgecolor": "black", "linewidth": 0.5}
    bar_opts = {**bar_defaults, **(bar_kwargs or {})}

    if multi_roi:
        roi_order = roi_order or sorted(roi_records.keys())
        roi_delta_frames: dict[str, pd.DataFrame] = {}
        all_deltas: list[float] = []
        all_epochs: set[int] = set()

        for roi in roi_order:
            df_roi = pd.DataFrame(roi_records.get(roi, []))
            if not df_roi.empty:
                df_roi = df_roi.sort_values("perturbation_epoch")
                all_deltas.extend(df_roi["delta_neural_alignment"].tolist())
                all_epochs.update(df_roi["perturbation_epoch"].tolist())
            roi_delta_frames[roi] = df_roi

        if not all_deltas:
            raise ValueError(
                "No overlapping epochs between the baseline and perturbed runs for the specified ROIs."
            )

        global_ymin = min(all_deltas)
        global_ymax = max(all_deltas)
        if math.isclose(global_ymin, global_ymax):
            pad = 0.05 if math.isclose(global_ymin, 0.0) else abs(global_ymin) * 0.05
            global_ymin -= pad
            global_ymax += pad
        else:
            pad = 0.05 * (global_ymax - global_ymin)
            global_ymin -= pad
            global_ymax += pad

        xticks = sorted(all_epochs)
        n_rois = len(roi_order)
        n_cols = min(4, n_rois)
        n_rows = math.ceil(n_rois / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if isinstance(axes, plt.Axes):
            axes_flat = [axes]
        elif hasattr(axes, "flat"):
            axes_flat = list(axes.flat)
        else:
            axes_flat = [axes]

        for idx, roi in enumerate(roi_order):
            ax_roi = axes_flat[idx]
            df_roi = roi_delta_frames.get(roi, pd.DataFrame())
            if not df_roi.empty:
                colors = [
                    positive_color
                    if val > 0
                    else negative_color
                    if val < 0
                    else zero_color
                    for val in df_roi["delta_neural_alignment"]
                ]
                ax_roi.bar(
                    df_roi["perturbation_epoch"],
                    df_roi["delta_neural_alignment"],
                    color=colors,
                    **bar_opts,
                )
                ax_roi.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
                if xticks:
                    ax_roi.set_xticks(xticks)
                if xticks:
                    ax_roi.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
                ax_roi.set_ylim(global_ymin, global_ymax)
                ax_roi.set_xlabel("Perturbation Epoch", fontweight="bold", fontsize=11)
                ax_roi.set_ylabel(r"$\Delta$ Neural Alignment", fontweight="bold", fontsize=11)
                ax_roi.set_title(roi, fontweight="bold", fontsize=14)
                ax_roi.grid(True, linestyle="--", axis="y", alpha=0.3)
                ax_roi.tick_params(axis="both", labelsize=10)
                ax_roi.spines["top"].set_visible(False)
                ax_roi.spines["right"].set_visible(False)
            else:
                ax_roi.text(
                    0.5,
                    0.5,
                    f"No data for {roi}",
                    transform=ax_roi.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax_roi.set_axis_off()

        for idx in range(n_rois, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        if title:
            fig.suptitle(title, fontweight="bold", fontsize=16, y=0.98)
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
        else:
            fig.tight_layout()

        return fig, roi_delta_frames

    if not records:
        raise ValueError(
            "No overlapping epochs between the baseline and perturbed runs. "
            "Ensure that the inputs contain matching epoch values."
        )

    delta_df = pd.DataFrame(records).sort_values("perturbation_epoch")
    ax = ax or plt.gca()

    colors: Sequence[str] = [
        positive_color if delta > 0 else negative_color if delta < 0 else zero_color
        for delta in delta_df["delta_neural_alignment"]
    ]

    ax.bar(
        delta_df["perturbation_epoch"],
        delta_df["delta_neural_alignment"],
        color=colors,
        **bar_opts,
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.8)
    ax.grid(True, linestyle="--", axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Perturbation Epoch", fontweight="bold", fontsize=14)
    ax.set_ylabel(r"$\Delta$ Neural Alignment (mean ROI)", fontweight="bold", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    return ax, delta_df.reset_index(drop=True)

