from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd


def load_training_results(
    csv_path: Union[str, Path],
    *,
    truncate_at_min_test_loss: bool = True,
    multiple_files: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load training-results data and optionally keep rows through the min test-loss epoch.

    Parameters
    ----------
    csv_path:
        Path to a training-results CSV file or, when `multiple_files=True`, a directory
        containing multiple CSVs (such as the single-epoch perturbation sweep outputs).
    truncate_at_min_test_loss:
        When True, rows following the minimum `test_loss` epoch are discarded.
    multiple_files:
        When True, all CSVs under `csv_path` (searched recursively) are loaded and a
        dictionary keyed by each file's stem is returned.

    Returns
    -------
    Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        A single DataFrame for `multiple_files=False`, otherwise a dictionary of
        DataFrames keyed by filename stem.
    """

    path = Path(csv_path)

    def _truncate(df: pd.DataFrame) -> pd.DataFrame:
        if not truncate_at_min_test_loss:
            return df
        if "test_loss" not in df.columns:
            raise ValueError(
                "Column 'test_loss' is required when truncate_at_min_test_loss=True."
            )
        if df.empty:
            return df.copy()
        min_idx = df["test_loss"].idxmin()
        return df.loc[:min_idx].copy()

    if multiple_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Training results directory not found: {path.resolve()}"
            )
        if not path.is_dir():
            raise ValueError("`csv_path` must be a directory when multiple_files=True.")

        csv_files = sorted(path.rglob("training_res*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files starting with 'training_res' and ending with '.csv' were found under directory: {path.resolve()}"
            )

        data: Dict[str, pd.DataFrame] = {}
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data[csv_file.stem] = _truncate(df)
        return data

    if not path.exists():
        raise FileNotFoundError(f"Training results file not found: {path.resolve()}")
    if not path.is_file():
        raise ValueError("`csv_path` must be a file when multiple_files=False.")

    df = pd.read_csv(path)
    return _truncate(df)