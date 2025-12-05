from __future__ import annotations

import re
from typing import Callable, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_single_epoch_delta_test_loss"]

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


def plot_single_epoch_delta_test_loss(
    baseline_df: pd.DataFrame,
    perturbed_runs: Mapping[str, pd.DataFrame],
    *,
    perturbation_epochs: Mapping[str, int] | None = None,
    epoch_parser: EpochParser | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    positive_color: str = "#c0392b",
    negative_color: str = "#27ae60",
    zero_color: str = "#7f8c8d",
    bar_kwargs: Mapping[str, float] | None = None,
) -> tuple[plt.Axes, pd.DataFrame]:
    """
    Plot the perturbation-induced delta in test loss for single-epoch sweeps.

    Parameters
    ----------
    baseline_df:
        DataFrame containing the unperturbed training metrics with columns
        `epoch` and `test_loss`.
    perturbed_runs:
        Mapping from run identifier to DataFrame that includes `epoch` and
        `test_loss`. Each run corresponds to a perturbation applied at a single
        epoch.
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

    Returns
    -------
    (matplotlib.axes.Axes, pandas.DataFrame)
        The rendered axis and a DataFrame summarizing the delta test loss per
        perturbation epoch.
    """

    plt.rcParams["font.family"] = "Times New Roman"

    required_cols = {"epoch", "test_loss"}
    missing = required_cols - set(baseline_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Baseline DataFrame missing required columns: {missing_str}"
        )

    epoch_lookup: MutableMapping[str, int] = (
        {k: int(v) for k, v in (perturbation_epochs or {}).items()}
    )
    parser = epoch_parser or _infer_epoch_from_name

    records: list[dict[str, float | int | str]] = []

    for run_name, df in perturbed_runs.items():
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

        run_point = df[df["epoch"] == perturb_epoch]
        baseline_point = baseline_df[baseline_df["epoch"] == perturb_epoch]

        if run_point.empty or baseline_point.empty:
            continue

        run_loss = float(run_point["test_loss"].iloc[0])
        baseline_loss = float(baseline_point["test_loss"].iloc[0])
        delta = run_loss - baseline_loss

        records.append(
            {
                "run_name": run_name,
                "perturbation_epoch": perturb_epoch,
                "delta_test_loss": delta,
                "perturbed_test_loss": run_loss,
                "baseline_test_loss": baseline_loss,
            }
        )

    if not records:
        raise ValueError(
            "No overlapping epochs between the baseline and perturbed runs. "
            "Ensure that the inputs contain matching epoch values."
        )

    delta_df = pd.DataFrame(records).sort_values("perturbation_epoch")
    ax = ax or plt.gca()

    bar_defaults = {"alpha": 0.7, "edgecolor": "black", "linewidth": 0.5}
    bar_opts = {**bar_defaults, **(bar_kwargs or {})}

    colors: Sequence[str] = [
        positive_color if delta > 0 else negative_color if delta < 0 else zero_color
        for delta in delta_df["delta_test_loss"]
    ]

    ax.bar(
        delta_df["perturbation_epoch"],
        delta_df["delta_test_loss"],
        color=colors,
        **bar_opts,
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.8)
    ax.grid(True, linestyle="--", axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Perturbation Epoch", fontweight="bold", fontsize=14)
    ax.set_ylabel(r"$\Delta$ Test Loss", fontweight="bold", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    return ax, delta_df.reset_index(drop=True)

