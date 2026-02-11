from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rirmega_eval.eval.harness import load_official_split
from rirmega_eval.io.hf import load_audio_ref
from rirmega_eval.io.parquet import read_metadata_parquet
from rirmega_eval.utils.seed import seed_everything

from baselines.baseline1_ml.features import FeatureConfig, extract_features, vectorize
from baselines.baseline1_ml.model import TARGETS, ModelConfig, make_model


def load_cfg(path: Path) -> tuple[ModelConfig, FeatureConfig]:
    d = json.loads(path.read_text(encoding="utf-8"))
    mc = ModelConfig(**d["model"])
    fc = FeatureConfig(**d["features"])
    return mc, fc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--task", type=str, default="v1_param_estimation")
    ap.add_argument("--config", type=Path, default=Path("baselines/configs/baseline1_default.json"))
    ap.add_argument("--out_dir", type=Path, default=Path("baselines/baseline1_ml/out"))
    args = ap.parse_args()

    seed_everything(1337)
    mc, fc = load_cfg(args.config)

    meta = read_metadata_parquet(args.dataset_dir / "data" / "metadata" / "rir_metadata.parquet")
    meta = meta[meta["view"] == args.task].copy()
    meta_idx = meta.set_index("sample_id")

    train_ids = load_official_split(args.dataset_dir, task=args.task, split="train")
    dev_ids = load_official_split(args.dataset_dir, task=args.task, split="dev")

    # Feature keys (fixed order)
    feature_keys = (
        "loge_mean",
        "loge_std",
        "loge_p95",
        "loge_p05",
        "decay_slope_db_per_s",
        "decay_intercept_db",
        "spec_centroid_hz",
        "spec_bandwidth_hz",
    )

    def build_xy(ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        Xs, Ys, kept = [], [], []
        for sid in ids:
            r = meta_idx.loc[sid]
            x, fs = load_audio_ref(r["rir_ref"], fallback_fs=int(r.get("fs", 48000)))
            feats = extract_features(x, fs, fc)
            Xs.append(vectorize(feats, feature_keys))
            Ys.append(np.array([float(r[t]) for t in TARGETS], dtype=np.float32))
            kept.append(sid)
        return np.stack(Xs), np.stack(Ys), kept

    Xtr, Ytr, _ = build_xy(train_ids)
    Xdv, Ydv, dev_kept = build_xy(dev_ids)

    model = make_model(mc)
    model.fit(Xtr, Ytr)
    Yhat = model.predict(Xdv).astype(np.float32)

    # Save dev predictions for official evaluator
    pred = pd.DataFrame({"sample_id": dev_kept})
    for i, t in enumerate(TARGETS):
        pred[t] = Yhat[:, i]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "dev_predictions.parquet"
    pred.to_parquet(pred_path, index=False)

    # Snapshot a simple reference metric report (filled by official evaluator in practice)
    snapshot = {
        "note": "Run scripts/evaluate.py on dev_predictions.parquet to record official baseline numbers.",
        "config": json.loads(args.config.read_text(encoding="utf-8")),
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote: {pred_path}")


if __name__ == "__main__":
    main()

