from __future__ import annotations

import argparse
import json
from pathlib import Path

from rirmega_eval.eval.harness import evaluate_predictions_file, validate_predictions_file
from rirmega_eval.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=False, default=Path("./eval_out"))
    ap.add_argument("--validate_only", action="store_true")
    ap.add_argument("--log_level", type=str, default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    if args.validate_only:
        validate_predictions_file(args.predictions, task=args.task)
        print("Predictions schema: OK")
        return

    out = evaluate_predictions_file(
        dataset_dir=args.dataset_dir,
        task=args.task,
        split=args.split,
        predictions_path=args.predictions,
        out_dir=args.out_dir,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

