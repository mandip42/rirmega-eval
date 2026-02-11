from __future__ import annotations

from pathlib import Path

from rirmega_eval.utils.hashing import sha256_file


def write_sha256sums(root_dir: Path, out_path: Path) -> None:
    """
    Write sha256 checksums for all files under root_dir (excluding the out file itself).
    Output format: '<sha256>  <relative/path>' with forward slashes.
    """
    root_dir = root_dir.resolve()
    out_path = out_path.resolve()

    files: list[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() == out_path:
            continue
        files.append(p)

    files.sort(key=lambda x: str(x.relative_to(root_dir)).replace("\\", "/"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for p in files:
            rel = str(p.relative_to(root_dir)).replace("\\", "/")
            digest = sha256_file(p)
            f.write(f"{digest}  {rel}\n")
