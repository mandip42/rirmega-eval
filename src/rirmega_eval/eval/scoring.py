from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS: list[str] = ["rt60_s", "edt_s", "drr_db", "c50_db", "c80_db", "ts_s"]



def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.nanmean(np.abs(y_true - y_pred))
    return float(m)


def score_param_estimation(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """
    Scores parameter estimation predictions.

    This function is tolerant to partial target sets:
    it scores only targets that exist in BOTH gt and pred.
    """
    # inner join on sample_id
    merged = gt.merge(pred, on="sample_id", how="inner", suffixes=("_gt", "_pred"))

    # Determine targets to score from columns present in BOTH
    # Expect raw gt/pred columns like 'rt60_s', 'drr_db', etc (pre-merge).
    # After merge we have rt60_s_gt and rt60_s_pred.
    candidate_targets = []
    for c in gt.columns:
        if c == "sample_id":
            continue
        if c in pred.columns:
            candidate_targets.append(c)

    metrics: dict[str, float | int | None] = {}
    n_scored = int(len(merged))

    for t in candidate_targets:
        gt_col = f"{t}_gt"
        pred_col = f"{t}_pred"
        if gt_col not in merged.columns or pred_col not in merged.columns:
            continue

        y_true = merged[gt_col].to_numpy(dtype=np.float64)
        y_pred = merged[pred_col].to_numpy(dtype=np.float64)

        # if all-NaN, skip
        if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
            metrics[f"mae_{t}"] = None
            continue

        metrics[f"mae_{t}"] = mae(y_true, y_pred)

    # Overall score: mean of available MAEs (ignores None)
    vals = [v for v in metrics.values() if isinstance(v, float)]
    metrics["overall_score"] = float(np.mean(vals)) if vals else None
    metrics["n_scored"] = n_scored

    return metrics
