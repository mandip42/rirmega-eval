from __future__ import annotations

import numpy as np
import pandas as pd


from pathlib import Path
REQUIRED_COLS: list[str] = [
    "sample_id",
    "view",
    "source_dataset",
    "rir_id",
    "rir_ref",
    "fs",
    "room_id",
    "family",
    "rt60_s",
    "edt_s",
    "drr_db",
    "c50_db",
    "c80_db",
    "ts_s",
]


def validate_metadata_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist (fill with NaN or None)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan if c.endswith(("_s", "_db")) else None

    # Types
    df["sample_id"] = df["sample_id"].astype(str)
    df["view"] = df["view"].astype(str)
    df["source_dataset"] = df["source_dataset"].astype(str)
    df["rir_id"] = df["rir_id"].astype(object)
    df["fs"] = df["fs"].fillna(48000).astype(int)

    # Uniqueness
    if df["sample_id"].duplicated().any():
        dup = df[df["sample_id"].duplicated()]["sample_id"].iloc[0]
        raise ValueError(f"Duplicate sample_id detected: {dup}")

    return df




def validate_dataset_dir(dataset_dir: Path) -> None:
    # call your existing internal checks here
    # raise ValueError with clear message if invalid
    ...
