from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rirmega_eval.eval.scoring import TARGETS, score_param_estimation
from rirmega_eval.eval.slicing import compute_slices, slice_definitions
from rirmega_eval.io.parquet import read_metadata_parquet
from rirmega_eval.io.splits import read_split_file
from rirmega_eval.utils.json import write_json
import logging
logger = logging.getLogger(__name__)


def _submission_hash(metrics_json_path: Path, per_slice_csv_path: Path) -> str:
    h = hashlib.sha256()
    h.update(metrics_json_path.read_bytes())
    h.update(per_slice_csv_path.read_bytes())
    return h.hexdigest()


def load_official_split(dataset_dir: Path, task: str, split: str) -> list[str]:
    p = dataset_dir / "data" / "splits" / "v1" / task / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_split_file(str(p))




from pathlib import Path
from typing import Sequence

import pandas as pd


def _task_targets(task: str) -> tuple[list[str], list[str]]:
    """
    Returns (required_targets, optional_targets) for a task.

    v0.1.0 note:
    - RIR-Mega data-mini provides rt60/drr/c50/c80 in its metrics dict.
    - edt_s and ts_s are not available unless computed from waveform.
    """
    task = task.strip()

    if task == "v1_param_estimation":
        required = ["rt60_s", "drr_db", "c50_db", "c80_db"]
        optional = ["edt_s", "ts_s"]
        return required, optional

    # Default behavior for other tasks: require no numeric targets at validation time.
    # The scorer will compute what it can based on intersection of columns.
    return [], []


def validate_predictions_file(predictions_path: str | Path, *, task: str) -> pd.DataFrame:
    """
    Validate a submission/predictions file.

    Requirements:
    - Must be parquet or jsonl
    - Must include 'sample_id'
    - Must include at least one numeric prediction column for the task
      (for v1_param_estimation: at least one of rt60_s, drr_db, c50_db, c80_db, edt_s, ts_s)

    Returns the loaded dataframe (so caller can reuse it).
    """
    p = Path(predictions_path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {p}")

    if p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in [".jsonl"]:
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"Unsupported predictions format: {p.suffix}. Use .parquet or .jsonl")

    if "sample_id" not in df.columns:
        raise ValueError("Predictions must include a 'sample_id' column")

    df["sample_id"] = df["sample_id"].astype(str)

    required, optional = _task_targets(task)
    allowed = list(dict.fromkeys(required + optional))  # preserve order, unique

    if allowed:
        present = [c for c in allowed if c in df.columns]
        if len(present) == 0:
            raise ValueError(
                f"Predictions must include at least one target column for {task}. "
                f"Expected one of: {allowed}"
            )

        # Only enforce "required" if you truly want to.
        # For v0.1.0 we do NOT hard fail on missing required targets, because
        # some submissions may only predict a subset. We score intersection.
        # If you want strictness later, re-enable this:
        # missing_required = [c for c in required if c not in df.columns]
        # if missing_required:
        #     raise ValueError(f"Predictions missing required targets: {missing_required}")

    return df



def evaluate_predictions_file(
    dataset_dir: Path,
    task: str,
    split: str,
    predictions_path: Path,
    out_dir: Path,
) -> dict[str, object]:
    validate_predictions_file(predictions_path, task=task)
    meta = read_metadata_parquet(dataset_dir / "data" / "metadata" / "rir_metadata.parquet")
    meta = compute_slices(meta)

    ids = load_official_split(dataset_dir, task=task, split=split)
    gt = meta[(meta["view"] == task) & (meta["sample_id"].isin(ids))].copy()

    if predictions_path.suffix.lower() == ".parquet":
        pred = validate_predictions_file(predictions_path, task=task)

    else:
        pred = pd.read_json(predictions_path, lines=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "v1_param_estimation":
        available_targets = [t for t in TARGETS if t in pred.columns]
        missing_targets = [t for t in TARGETS if t not in pred.columns]

        if not available_targets:
            raise ValueError(
                f"No prediction targets found for {task}. Expected at least one of: {TARGETS}"
            )

        if missing_targets:
            logger.warning(
                "Predictions missing targets for %s: %s (scoring available only)",
                task,
                missing_targets,
            )

        gt_eval = gt[["sample_id", *available_targets]].copy()
        pred_eval = pred[["sample_id", *available_targets]].copy()

        metrics = score_param_estimation(gt=gt_eval, pred=pred_eval)

        # Write outputs
        metrics_path = out_dir / "metrics.json"
        write_json(metrics_path, metrics)

        # You can keep your existing slicing/worst-examples code if it expects columns.
        # For now, write empty placeholders to satisfy CLI outputs.
        per_slice_path = out_dir / "per_slice_metrics.csv"
        worst_path = out_dir / "worst_examples.csv"

        pd.DataFrame([]).to_csv(per_slice_path, index=False)
        pd.DataFrame([]).to_csv(worst_path, index=False)

        summary = {
            "task": task,
            "split": split,
            "n_gt": int(len(gt)),
            "n_pred": int(len(pred)),
            "outputs": {
                "metrics_json": str(metrics_path),
                "per_slice_metrics_csv": str(per_slice_path),
                "worst_examples_csv": str(worst_path),
            },
            "metrics": metrics,
        }

        summary_path = out_dir / "summary.json"
        write_json(summary_path, summary)
        return summary

    raise NotImplementedError(f"Task not implemented in evaluator yet: {task}")


