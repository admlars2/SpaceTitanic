from pathlib import Path

import numpy as np
import pandas as pd

from .settings import CABIN_BIN_SIZE


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until we find pyproject.toml (repo root)."""
    start = start or Path.cwd()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    return start


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    project_dir = find_project_root()
    data_dir = project_dir / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df, project_dir


def compute_group_sizes(df: pd.DataFrame) -> pd.Series:
    group_str = df["PassengerId"].astype("string").str.split("_").str[0]
    group_id = pd.to_numeric(group_str, errors="coerce")
    return group_id.value_counts(dropna=False)


def compute_group_modes(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    group_str = df["PassengerId"].astype("string").str.split("_").str[0]
    group_id = pd.to_numeric(group_str, errors="coerce")
    modes: dict[str, pd.Series] = {}

    for col in cols:
        if col in df.columns:
            series = df[col].astype("string")
        elif col == "CabinDeck":
            series = (
                df["Cabin"].astype("string").str.split("/", expand=True).get(0)
            )
        elif col == "CabinSide":
            series = (
                df["Cabin"].astype("string").str.split("/", expand=True).get(2)
            )
        else:
            series = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

        def mode_or_missing(values: pd.Series):
            clean = values.dropna()
            if clean.empty:
                return pd.NA
            mode = clean.mode()
            return mode.iloc[0] if not mode.empty else pd.NA

        modes[col] = series.groupby(group_id).agg(mode_or_missing)

    return modes

def compute_cabin_bins(df: pd.DataFrame, bin_size: int = CABIN_BIN_SIZE) -> np.ndarray:
    cabin_num = pd.to_numeric(
        df["Cabin"].astype("string").str.split("/", expand=True).get(1),
        errors="coerce",
    )
    if cabin_num.notna().any():
        max_num = int(cabin_num.max())
    else:
        max_num = bin_size
    end = max(bin_size, ((max_num // bin_size) + 1) * bin_size)
    return np.arange(0, end + bin_size, bin_size)