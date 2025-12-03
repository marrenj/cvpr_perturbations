from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def merge_behavioral_and_neural(
    behavioral_df: pd.DataFrame,
    neural_data: Union[pd.DataFrame, str, Path],
) -> pd.DataFrame:
    """
    Merge behavioral training results with neural RSA results on the `epoch` column.

    Parameters
    ----------
    behavioral_df:
        DataFrame containing the training metrics (must include an `epoch` column).
    neural_data:
        Either a DataFrame with neural RSA metrics or a path (str/Path) to a CSV file
        containing columns `epoch`, `roi`, `rho`, `p_value`, etc.

    Returns
    -------
    pd.DataFrame
        A left-joined DataFrame containing both behavioral and neural metrics.
    """
    if "epoch" not in behavioral_df.columns:
        raise ValueError("`behavioral_df` must include an 'epoch' column.")

    if isinstance(neural_data, pd.DataFrame):
        neural_df = neural_data.copy()
    else:
        path = Path(neural_data)
        if not path.exists():
            raise FileNotFoundError(f"Neural results file not found: {path.resolve()}")
        neural_df = pd.read_csv(path)

    if "epoch" not in neural_df.columns:
        raise ValueError("`neural_data` must include an 'epoch' column.")

    merged = pd.merge(behavioral_df, neural_df, on="epoch", how="left")
    return merged