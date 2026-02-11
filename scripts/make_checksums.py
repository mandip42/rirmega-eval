from __future__ import annotations

import argparse
from pathlib import Path

from rirmega_eval.utils.hashing import sha256_file


def checksums_main(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    out = root / "checks" / "sha256sums.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    targets = [
        root / "manifest.json",
        root / "data" / "metadata" / "rir_metadata.parquet",
    ]

    # Split files (recursive)
    splits_root = root / "data" / "splits"
    if splits_root.exists():
        for p in sorted(splits_root.rglob("*.txt")):
            targets.append(p)

    lines = []
    for p in targets:
        rel = p.relative_to(root).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checksums: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    args = ap.parse_args()
    checksums_main(args.dataset_dir)


if __name__ == "__main__":
    main()

