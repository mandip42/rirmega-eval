
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rirmega_eval.eval.harness import load_official_split
from rirmega_eval.io.hf import load_audio_ref
from rirmega_eval.io.parquet import read_metadata_parquet
from rirmega_eval.metrics.canonical import compute_all_metrics
from rirmega_eval.utils.seed import seed_everything


def run(dataset_dir: Path, split: str, out_path: Path) -> None:
    seed_everything(1337)
    meta = read_metadata_parquet(dataset_dir / "data" / "metadata" / "rir_metadata.parquet")
    sample_ids = load_official_split(dataset_dir, task="v1_param_estimation", split=split)

    rows = []
    meta_idx = meta.set_index("sample_id")
    for sid in sample_ids:
        r = meta_idx.loc[sid]
        x, fs = load_audio_ref(r["rir_ref"], fallback_fs=int(r.get("fs", 48000)))
        m = compute_all_metrics(x, fs=fs)
        rows.append({"sample_id": sid, **m})

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote baseline predictions: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.dataset_dir, args.split, args.out)


if __name__ == "__main__":
    main()

