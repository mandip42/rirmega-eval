from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits_grouped(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    ratios: tuple[float, float, float],
    require_unseen_group_test: bool,
) -> dict[str, list[str]]:
    """
    Deterministic grouped split.
    If group_col missing or all null, falls back to iid split.

    Ratios are (train, dev, test). Uses seed=1337 by default.
    """
    rng = np.random.default_rng(seed)
    ids = df["sample_id"].tolist()

    g = df[group_col] if group_col in df.columns else None
    if g is None or g.isna().all():
        idx = np.arange(len(ids))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(ratios[0] * n))
        n_dv = int(round(ratios[1] * n))
        tr = [ids[i] for i in idx[:n_tr]]
        dv = [ids[i] for i in idx[n_tr : n_tr + n_dv]]
        te = [ids[i] for i in idx[n_tr + n_dv :]]
        return {"train": tr, "dev": dv, "test": te}

    # Grouped split
    df2 = df.copy()
    df2[group_col] = df2[group_col].fillna("__NO_GROUP__")
    groups = sorted(df2[group_col].unique().tolist())
    rng.shuffle(groups)

    n = len(groups)
    n_tr = int(round(ratios[0] * n))
    n_dv = int(round(ratios[1] * n))
    g_tr = set(groups[:n_tr])
    g_dv = set(groups[n_tr : n_tr + n_dv])
    g_te = set(groups[n_tr + n_dv :])

    if require_unseen_group_test and "__NO_GROUP__" in g_te and n > 1:
        # Move NO_GROUP out of test if possible to preserve unseen-room semantics
        g_te.remove("__NO_GROUP__")
        g_dv.add("__NO_GROUP__")

    tr = df2[df2[group_col].isin(g_tr)]["sample_id"].tolist()
    dv = df2[df2[group_col].isin(g_dv)]["sample_id"].tolist()
    te = df2[df2[group_col].isin(g_te)]["sample_id"].tolist()
    return {"train": tr, "dev": dv, "test": te}

def read_split_file(path: str) -> list[str]:
    import pathlib

    p = pathlib.Path(path)
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

