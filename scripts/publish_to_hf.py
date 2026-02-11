from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import create_repo, upload_file, upload_folder

LOG = logging.getLogger("publish_to_hf")


HF_CARD_TEMPLATE = """---
license: cc-by-nc-4.0
task_categories:
- audio-to-audio
- audio-classification
tags:
- rir
- acoustics
- benchmark
- evaluation
---

# RIRMega-Eval

Official benchmark views and evaluation harness artifacts built from:
- `mandipgoswami/rirmega` :contentReference[oaicite:6]{index=6}
- `mandipgoswami/rir-mega-speech` :contentReference[oaicite:7]{index=7}

This HF repo stores:
- metadata parquet
- fixed splits
- checksums + manifest
- (optional) a small core audio subset

## Tasks
- `v1_param_estimation`: RIR -> RT60, EDT, DRR, C50, C80, Ts
- `v1_auralization_consistency`: (dry + RIR) -> convolved comparisons

## How to evaluate
Use the official evaluator:
- Code: https://github.com/mandip42/rirmega-eval
- CLI: `python scripts/evaluate.py ...`

## Citation
See `CITATION.cff` in the code repo. Add Zenodo DOI when minted.
"""


def _collect_upload_paths(dataset_dir: Path, upload_core_audio: bool) -> list[Path]:
    paths = [
        dataset_dir / "manifest.json",
        dataset_dir / "checks" / "sha256sums.txt",
        dataset_dir / "data" / "metadata" / "rir_metadata.parquet",
        dataset_dir / "data" / "splits",
    ]
    if upload_core_audio:
        audio_dir = dataset_dir / "data" / "audio"
        if audio_dir.exists():
            paths.append(audio_dir)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--hf_dataset_id", type=str, required=True)
    ap.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--upload_core_audio", action="store_true")
    args = ap.parse_args()

    token = os.environ.get(args.hf_token_env)
    if not token:
        raise RuntimeError(f"Missing HF token. Set env var {args.hf_token_env}.")



    # Create repo if needed
    create_repo(
        repo_id=args.hf_dataset_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # Upload artifacts
    upload_paths = _collect_upload_paths(args.dataset_dir, args.upload_core_audio)
    for p in upload_paths:
        if p.is_dir():
            upload_folder(
                repo_id=args.hf_dataset_id,
                repo_type="dataset",
                folder_path=str(p),
                path_in_repo=str(p.relative_to(args.dataset_dir)).replace("\\", "/"),
                token=token,
                commit_message=f"Upload {p.name}",
            )
        else:
            upload_file(
                repo_id=args.hf_dataset_id,
                repo_type="dataset",
                path_or_fileobj=str(p),
                path_in_repo=str(p.relative_to(args.dataset_dir)).replace("\\", "/"),
                token=token,
                commit_message=f"Upload {p.name}",
            )

    # Upload dataset card
    card_path = args.dataset_dir / "HF_README.md"
    card_path.write_text(HF_CARD_TEMPLATE, encoding="utf-8")
    upload_file(
        repo_id=args.hf_dataset_id,
        repo_type="dataset",
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        token=token,
        commit_message="Update dataset card",
    )

    LOG.info("Published to HF dataset repo: %s", args.hf_dataset_id)
    print(f"Done: https://huggingface.co/datasets/{args.hf_dataset_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

