import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["plot_behavioral_alignment_heatmap", "plot_min_test_loss_heatmap"]


def plot_behavioral_alignment_heatmap(
    baseline_df: pd.DataFrame,
    run_metadata: list[dict],
    *,
    value_column: str = "behavioral_rsa_rho",
    exclude_start_epochs: list[int] | None = None,
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "RdYlGn",
    title: str | None = "Maximum Behavioral Alignment Deviations",
    annot: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Plot a heatmap of the deviation between each perturbation run's best behavioral
    alignment and the best baseline behavioral alignment.

    Parameters
    ----------
    baseline_df:
        DataFrame containing the baseline run with columns `epoch` and `value_column`.
    run_metadata:
        Iterable of dicts with keys: `run_name`, `df` (per-run DataFrame),
        `start_epoch`, and `length`.
    value_column:
        Name of the column containing the behavioral alignment metric.
    exclude_start_epochs:
        Optional list of start epochs to drop before plotting.
    figsize:
        Figure size passed to matplotlib.
    cmap:
        Colormap for the heatmap.
    title:
        Plot title. Set to None to omit.
    annot:
        Whether to annotate cells with numeric values.
    vmin, vmax:
        Color scale bounds passed to seaborn. Use the same vmin/vmax across calls
        to keep multiple heatmaps on a consistent scale.

    Returns
    -------
    fig, ax, deviation_df, heatmap_data, data_min, data_max
    """

    if value_column not in baseline_df.columns:
        raise ValueError(f"Baseline DataFrame missing column '{value_column}'.")

    baseline_max_ba = baseline_df[value_column].max()

    deviation_data: list[dict] = []
    for run_info in run_metadata:
        df = run_info.get("df")
        start_epoch = run_info.get("start_epoch")
        length = run_info.get("length")

        if df is None or df.empty or value_column not in df.columns:
            continue

        df_sorted = df.sort_values("epoch")
        max_ba = df_sorted[value_column].max()
        max_ba_epoch = df_sorted.loc[df_sorted[value_column].idxmax(), "epoch"]
        deviation = max_ba - baseline_max_ba

        deviation_data.append(
            {
                "run_name": run_info.get("run_name"),
                "start_epoch": start_epoch,
                "length": length,
                "deviation": deviation,
                "max_ba": max_ba,
                "max_ba_epoch": max_ba_epoch,
                "baseline_max_ba": baseline_max_ba,
            }
        )

    deviation_df = pd.DataFrame(deviation_data)
    if deviation_df.empty:
        raise ValueError("No data to plot (check inputs).")

    if exclude_start_epochs:
        deviation_df = deviation_df[~deviation_df["start_epoch"].isin(exclude_start_epochs)]
        if deviation_df.empty:
            raise ValueError("All rows were excluded by `exclude_start_epochs`.")

    heatmap_data = deviation_df.pivot_table(
        values="deviation",
        index="length",
        columns="start_epoch",
        aggfunc="first",
    )
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    data_min = float(heatmap_data.min().min())
    data_max = float(heatmap_data.max().max())

    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Deviation"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Start Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel("Perturbation Length", fontsize=16, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    return fig, ax, deviation_df, heatmap_data, data_min, data_max


def plot_min_test_loss_heatmap(
    baseline_df: pd.DataFrame,
    run_metadata: list[dict],
    *,
    value_column: str = "test_loss",
    exclude_start_epochs: list[int] | None = None,
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "RdYlGn_r",
    title: str | None = "Minimum Test Loss Deviations",
    annot: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Plot a heatmap of the deviation between each perturbation run's minimum test loss
    and the baseline minimum test loss.

    Parameters
    ----------
    baseline_df:
        DataFrame containing the baseline run with columns `epoch` and `value_column`.
    run_metadata:
        Iterable of dicts with keys: `run_name`, `df` (per-run DataFrame),
        `start_epoch`, and `length`.
    value_column:
        Name of the column containing the test loss metric.
    exclude_start_epochs:
        Optional list of start epochs to drop before plotting.
    figsize:
        Figure size passed to matplotlib.
    cmap:
        Colormap for the heatmap (default reversed so lower loss is green).
    title:
        Plot title. Set to None to omit.
    annot:
        Whether to annotate cells with numeric values.
    vmin, vmax:
        Color scale bounds passed to seaborn. Use the same vmin/vmax across calls
        to keep multiple heatmaps on a consistent scale.

    Returns
    -------
    fig, ax, deviation_df, heatmap_data, data_min, data_max
    """

    if value_column not in baseline_df.columns:
        raise ValueError(f"Baseline DataFrame missing column '{value_column}'.")

    baseline_min = baseline_df[value_column].min()

    deviation_data: list[dict] = []
    for run_info in run_metadata:
        df = run_info.get("df")
        start_epoch = run_info.get("start_epoch")
        length = run_info.get("length")

        if df is None or df.empty or value_column not in df.columns:
            continue

        df_sorted = df.sort_values("epoch")
        min_loss = df_sorted[value_column].min()
        min_loss_epoch = df_sorted.loc[df_sorted[value_column].idxmin(), "epoch"]
        deviation = min_loss - baseline_min

        deviation_data.append(
            {
                "run_name": run_info.get("run_name"),
                "start_epoch": start_epoch,
                "length": length,
                "deviation": deviation,
                "min_test_loss": min_loss,
                "min_test_loss_epoch": min_loss_epoch,
                "baseline_min_test_loss": baseline_min,
            }
        )

    deviation_df = pd.DataFrame(deviation_data)
    if deviation_df.empty:
        raise ValueError("No data to plot (check inputs).")

    if exclude_start_epochs:
        deviation_df = deviation_df[~deviation_df["start_epoch"].isin(exclude_start_epochs)]
        if deviation_df.empty:
            raise ValueError("All rows were excluded by `exclude_start_epochs`.")

    heatmap_data = deviation_df.pivot_table(
        values="deviation",
        index="length",
        columns="start_epoch",
        aggfunc="first",
    )
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    data_min = float(heatmap_data.min().min())
    data_max = float(heatmap_data.max().max())

    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Deviation"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Start Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel("Perturbation Length", fontsize=16, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    return fig, ax, deviation_df, heatmap_data, data_min, data_max