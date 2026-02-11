from __future__ import annotations

import argparse
from pathlib import Path

from rirmega_eval.io.parquet import read_metadata_parquet
from rirmega_eval.io.schema import validate_metadata_schema


def validate_main(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    p = root / "data" / "metadata" / "rir_metadata.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = read_metadata_parquet(p)
    df2 = validate_metadata_schema(df)
    # Ensure split files exist for v1 tasks
    splits_root = root / "data" / "splits" / "v1"
    for task in ["v1_param_estimation", "v1_auralization_consistency"]:
        for split in ["train", "dev", "test"]:
            sp = splits_root / task / f"{split}.txt"
            if not sp.exists():
                raise FileNotFoundError(sp)
    print(f"Dataset validation OK. Rows: {df2.shape[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    args = ap.parse_args()
    validate_main(args.dataset_dir)


if __name__ == "__main__":
    main()

