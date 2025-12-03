from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_training_results(
    csv_path: Union[str, Path],
    *,
    truncate_at_min_test_loss: bool = True,
) -> pd.DataFrame:
    """
    Load a training-results CSV and optionally keep rows through the min test-loss epoch.

    Parameters
    ----------
    csv_path:
        File system path to the CSV produced by the training loop.
    truncate_at_min_test_loss:
        When True, rows following the minimum `test_loss` epoch are discarded. This
        mimics the manual filtering performed in several analysis notebooks.

    Returns
    -------
    pd.DataFrame
        The loaded (and optionally truncated) training results.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training results file not found: {path.resolve()}")

    df = pd.read_csv(path)

    if truncate_at_min_test_loss:
        if "test_loss" not in df.columns:
            raise ValueError(
                "Column 'test_loss' is required when truncate_at_min_test_loss=True."
            )
        min_idx = df["test_loss"].idxmin()
        df = df.loc[:min_idx].copy()

    return df