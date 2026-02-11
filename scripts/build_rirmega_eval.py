from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rirmega_eval.config import DEFAULT_SEED
from rirmega_eval.io.hf import (
    HFSourceConfig,
    iter_rirmega_rows,
    iter_rirmega_speech_rows,
    maybe_download_core_audio,
)

from rirmega_eval.io.schema import validate_dataset_dir
from rirmega_eval.utils.checksums import write_sha256sums
from rirmega_eval.utils.json import write_json

logger = logging.getLogger("build_rirmega_eval")


# -----------------------------
# Helpers
# -----------------------------
def _parse_py_literal_dict(s: str | None) -> dict[str, Any]:
    """Parse python-literal dict strings (single quotes) safely."""
    if not s or not isinstance(s, str):
        return {}
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Views
# -----------------------------
def build_view_param_estimation(
    hf_rir: HFSourceConfig,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    View 1: RIR parameter estimation.
    Uses RIR-Mega data-mini metadata and parses the 'metrics' field.

    Targets populated (if present in source metrics dict):
      - rt60_s (from metrics['rt60'])
      - drr_db (from metrics['drr_db'])
      - c50_db (from metrics['c50_db'])
      - c80_db (from metrics['c80_db'])

    Targets not present in source metrics (left as NaN for now):
      - edt_s
      - ts_s

    If you later want EDT/Ts to be non-NaN, compute from waveform using canonical metrics.
    """
    rows_out: list[dict[str, Any]] = []
    n = 0

    for r in iter_rirmega_rows(hf_rir):
        # r is a dict from CSV: includes id, family, fs, wav, room_size, source, microphone, metrics, ...
        rir_id = str(r.get("id", "")).strip()
        if not rir_id:
            continue

        m = _parse_py_literal_dict(r.get("metrics"))

        row = {
            "view": "v1_param_estimation",
            "sample_id": rir_id,  # stable benchmark key
            "rir_id": rir_id,
            "room_id": str(r.get("seed_room", "")) if r.get("seed_room") is not None else None,
            "room_class": str(r.get("family", "")) if r.get("family") is not None else None,
            "fs_hz": int(r.get("fs")) if r.get("fs") is not None else None,
            "wav_ref": str(r.get("wav", "")) if r.get("wav") is not None else None,
            "split_hint": str(r.get("split", "")) if r.get("split") is not None else None,
            # targets (filled from metrics dict)
            "rt60_s": _to_float(m.get("rt60")),
            "edt_s": None,  # not available in source metrics
            "drr_db": _to_float(m.get("drr_db")),
            "c50_db": _to_float(m.get("c50_db")),
            "c80_db": _to_float(m.get("c80_db")),
            "ts_s": None,  # not available in source metrics
        }

        rows_out.append(row)
        n += 1
        if max_rows is not None and n >= max_rows:
            break

    df = pd.DataFrame(rows_out)

    # enforce types for join keys
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)

    return df


def build_view_auralization_consistency(
    hf_speech: HFSourceConfig,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    View 2: Auralization consistency.
    Uses rir-mega-speech metadata/metadata.csv.

    We keep it schema-light here: store references and IDs needed for evaluation.
    Your evaluator can load the audio by reference if you later decide to include core audio.
    """
    rows_out: list[dict[str, Any]] = []
    n = 0

    for r in iter_rirmega_speech_rows(hf_speech):
        # We don't assume exact field names beyond common ones.
        # Store best-effort keys; missing fields remain None.
        clean_id = str(
            r.get("clean_id")
            or r.get("id")
            or r.get("sample_id")
            or r.get("utterance_id")
            or ""
        ).strip()
        if not clean_id:
            continue

        rir_id = str(r.get("rir_id") or "").strip()
        split = str(r.get("split") or "").strip()

        # Globally unique benchmark key
        sample_id = f"v2::{clean_id}::rir::{rir_id}::split::{split}"

        row = {
            "view": "v1_auralization_consistency",
            "sample_id": sample_id,
            "rir_id": str(r.get("rir_id") or "").strip() or None,
            "room_id": None,  # not provided in this metadata
            "room_class": None,  # not provided in this metadata
            "fs_hz": None,  # not provided (store later if you add)
            "dry_ref": None,  # audio paths not in metadata (you have 'audio' field)
            "wet_ref": r.get("audio"),  # treat 'audio' as the convolved/wet reference
            "rir_ref": None,
            "speaker_id": None,
            "utt_id": sample_id,
            "snr_db": None,
            # Optional direct targets from speech metadata (nice to keep)
            "rt60_s": _to_float(r.get("rt60")),
            "drr_db": _to_float(r.get("drr")),
            "c50_db": _to_float(r.get("c50")),
            "lufs": _to_float(r.get("lufs")),
            "duration_s": _to_float(r.get("duration_s")),
            "split_hint": str(r.get("split") or "").strip() or None,
        }

        rows_out.append(row)
        n += 1
        if max_rows is not None and n >= max_rows:
            break

    df = pd.DataFrame(rows_out)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


# -----------------------------
# Splits
# -----------------------------
@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.82
    dev: float = 0.087
    test: float = 0.093
    seed: int = DEFAULT_SEED


def _write_split_files(
    out_dir: Path,
    *,
    task_name: str,
    splits: dict[str, list[str]],
) -> None:
    split_dir = out_dir / "data" / "splits" / "v1" / task_name
    _ensure_dir(split_dir)

    for split_name, ids in splits.items():
        p = split_dir / f"{split_name}.txt"
        # stable, newline-delimited
        p.write_text("\n".join(ids) + "\n", encoding="utf-8")

def _make_grouped_splits(
    ids: list[str],
    groups: list[str],
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    """
    Deterministic grouped split: all items with the same group go to the same split.
    Uses shuffled unique groups then assigns groups to train/dev/test by ratio.
    """
    assert len(ids) == len(groups)

    # map group -> list of ids
    g2ids: dict[str, list[str]] = {}
    for sid, g in zip(ids, groups, strict=False):
        g2ids.setdefault(g, []).append(sid)

    uniq_groups = list(g2ids.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_groups)

    # assign groups
    n = len(uniq_groups)
    n_train = int(round(train_ratio * n))
    n_dev = int(round(dev_ratio * n))
    n_test = n - n_train - n_dev

    train_g = set(uniq_groups[:n_train])
    dev_g = set(uniq_groups[n_train : n_train + n_dev])
    test_g = set(uniq_groups[n_train + n_dev :])

    def collect(gs: set[str]) -> list[str]:
        out: list[str] = []
        for g in gs:
            out.extend(g2ids[g])
        return out

    train_ids = collect(train_g)
    dev_ids = collect(dev_g)
    test_ids = collect(test_g)

    # stable order for files
    train_ids.sort()
    dev_ids.sort()
    test_ids.sort()

    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def make_splits_for_view(
    df_view: pd.DataFrame,
    *,
    group_key: str | None,
    spec: SplitSpec,
) -> dict[str, list[str]]:
    """
    Deterministic train/dev/test split generation.
    If group_key is provided and exists, uses grouped splits for 'unseen room' behavior.
    """
    ids = df_view["sample_id"].astype(str).tolist()

    if group_key and group_key in df_view.columns:
        groups = df_view[group_key].astype(str).fillna("NA").tolist()
        return _make_grouped_splits(
            ids=ids,
            groups=groups,
            seed=spec.seed,
            train_ratio=spec.train,
            dev_ratio=spec.dev,
            test_ratio=spec.test,
        )

        return splits

    # fallback: simple deterministic shuffle split
    rng = np.random.default_rng(spec.seed)
    ids_arr = np.array(ids, dtype=object)
    rng.shuffle(ids_arr)

    n = len(ids_arr)
    n_train = int(round(spec.train * n))
    n_dev = int(round(spec.dev * n))
    n_test = n - n_train - n_dev

    train_ids = ids_arr[:n_train].tolist()
    dev_ids = ids_arr[n_train : n_train + n_dev].tolist()
    test_ids = ids_arr[n_train + n_dev :].tolist()

    return {"train": train_ids, "dev": dev_ids, "test": test_ids}



# -----------------------------
# Manifest
# -----------------------------
def _build_manifest(
    *,
    out_dir: Path,
    df_all: pd.DataFrame,
    hf_rir: HFSourceConfig,
    hf_speech: HFSourceConfig,
    build_kind: str,
) -> dict[str, Any]:
    views = sorted(df_all["view"].unique().tolist())
    counts_by_view = df_all.groupby("view")["sample_id"].count().to_dict()

    m = {
        "benchmark_id": "RIRMega-Eval",
        "version": "0.1.0",
        "build_kind": build_kind,
        "seed": DEFAULT_SEED,
        "sources": {
            "rirmega": asdict(hf_rir),
            "rirmega_speech": asdict(hf_speech),
        },
        "views": views,
        "counts_by_view": counts_by_view,
        "paths": {
            "metadata_parquet": "data/metadata/rir_metadata.parquet",
            "splits_root": "data/splits/v1/",
            "checksums": "checks/sha256sums.txt",
            "manifest": "manifest.json",
        },
    }
    return m


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RIRMega-Eval benchmark artifacts.")
    p.add_argument("--out", type=str, required=True, help="Output directory for built benchmark artifacts.")
    p.add_argument(
        "--build",
        type=str,
        default="full",
        choices=["core", "full"],
        help="core exports an optional small audio subset; full is metadata-only refs.",
    )
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level (INFO, DEBUG, ...).")
    p.add_argument("--max_rir_rows", type=int, default=0, help="Max RIR-Mega rows (0 = no limit).")
    p.add_argument("--max_speech_rows", type=int, default=0, help="Max RIR-Mega-Speech rows (0 = no limit).")
    p.add_argument("--core_audio", action="store_true", help="Enable core audio export (if implemented).")
    p.add_argument("--core_audio_max_items", type=int, default=0, help="Max core audio items to export.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    max_rir_rows = args.max_rir_rows if args.max_rir_rows and args.max_rir_rows > 0 else None
    max_speech_rows = args.max_speech_rows if args.max_speech_rows and args.max_speech_rows > 0 else None

    logger.info("Building RIRMega-Eval (%s) into %s", args.build, out_dir)

    # HF sources (pinned file paths for robustness)
    hf_rir = HFSourceConfig(
        dataset_id="mandipgoswami/rirmega",
        metadata_path="data-mini/metadata/metadata.csv",
    )
    hf_speech = HFSourceConfig(
        dataset_id="mandipgoswami/rir-mega-speech",
        metadata_path="metadata/metadata.csv",
    )

    # Build views
    df1 = build_view_param_estimation(hf_rir, max_rows=max_rir_rows)
    df2 = build_view_auralization_consistency(hf_speech, max_rows=max_speech_rows)

    df_all = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_all["sample_id"] = df_all["sample_id"].astype(str)

    # Write metadata parquet
    meta_dir = out_dir / "data" / "metadata"
    _ensure_dir(meta_dir)
    meta_path = meta_dir / "rir_metadata.parquet"
    df_all.to_parquet(meta_path, index=False)

    # Splits (v1)
    spec = SplitSpec(seed=DEFAULT_SEED)

    # View 1: group by room_id if available for unseen room holdout behavior
    v1 = df_all[df_all["view"] == "v1_param_estimation"].copy()
    splits_v1 = make_splits_for_view(v1, group_key="room_id", spec=spec)
    _write_split_files(out_dir, task_name="v1_param_estimation", splits=splits_v1)

    # View 2: also group by room_id when possible (best-effort)
    v2 = df_all[df_all["view"] == "v1_auralization_consistency"].copy()
    splits_v2 = make_splits_for_view(v2, group_key="room_id", spec=spec)
    _write_split_files(out_dir, task_name="v1_auralization_consistency", splits=splits_v2)

    # Optional core audio export
    if args.build == "core":
        out_audio_dir = out_dir / "data" / "audio"
        _ensure_dir(out_audio_dir)

        # Export core audio only if user requests it and max_items > 0
        maybe_download_core_audio(
            enable=args.core_audio,
            dataset_id=hf_rir.dataset_id,
            out_audio_dir=out_audio_dir,
            rows=v1.to_dict("records"),
            audio_key="wav_ref",
            revision=hf_rir.revision,
            max_items=int(args.core_audio_max_items),
        )

    # Manifest
    manifest = _build_manifest(
        out_dir=out_dir,
        df_all=df_all,
        hf_rir=hf_rir,
        hf_speech=hf_speech,
        build_kind=args.build,
    )
    write_json(out_dir / "manifest.json", manifest)

    # Validate (package-level)
    validate_dataset_dir(out_dir)

    # Checksums
    write_sha256sums(root_dir=out_dir, out_path=out_dir / "checks" / "sha256sums.txt")

    logger.info("Done. Artifacts written under %s", out_dir)


if __name__ == "__main__":
    main()

