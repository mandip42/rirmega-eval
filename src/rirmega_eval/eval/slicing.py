from __future__ import annotations

import pandas as pd


def add_rt60_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 10.0]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.7", "0.7-1.0", "1.0-1.5", "1.5-2.5", "2.5+"]
    df["rt60_bin"] = pd.cut(df["rt60_s"], bins=bins, labels=labels, include_lowest=True)
    return df


def slice_definitions() -> list[tuple[str, str]]:
    # (slice_name, column)
    return [
        ("family", "family"),
        ("rt60_bin", "rt60_bin"),
    ]


def compute_slices(meta: pd.DataFrame) -> pd.DataFrame:
    m = add_rt60_bin(meta)
    return m

