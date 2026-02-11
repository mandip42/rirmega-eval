from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HFSourceConfig:
    dataset_id: str
    split: str | None = "train"        # kept for backward compatibility
    streaming: bool = True             # kept for backward compatibility
    revision: str | None = None
    metadata_path: str | None = None   # path inside dataset repo


def _iter_csv_rows(local_csv_path: Path) -> Iterator[dict]:
    with local_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def iter_rirmega_rows(cfg: HFSourceConfig) -> Iterator[dict]:
    """
    Robust iterator for RIR-Mega rows.

    We intentionally avoid datasets.load_dataset(folder_based_builder) because
    the repo may contain multiple metadata.csv files with different features,
    which breaks HF's folder builder.
    """
    metadata_path = cfg.metadata_path or "data-mini/metadata/metadata.csv"

    logger.info(
        "Loading RIR-Mega metadata via hf_hub_download: %s@%s (%s)",
        cfg.dataset_id,
        cfg.revision,
        metadata_path,
    )

    local_path = Path(
        hf_hub_download(
            repo_id=cfg.dataset_id,
            repo_type="dataset",
            filename=metadata_path,
            revision=cfg.revision,
        )
    )

    yield from _iter_csv_rows(local_path)
def iter_rirmega_speech_rows(cfg: HFSourceConfig) -> Iterator[dict]:
    """
    Robust iterator for RIR-Mega-Speech rows.

    Same rationale as iter_rirmega_rows: avoid HF folder-based builder when the
    dataset contains multiple metadata files with differing schemas.
    """
    # Default path. If your rir-mega-speech repo uses a different location,
    # we will adjust once we see the actual file layout.
    metadata_path = cfg.metadata_path or "metadata/metadata.csv"


    logger.info(
        "Loading RIR-Mega-Speech metadata via hf_hub_download: %s@%s (%s)",
        cfg.dataset_id,
        cfg.revision,
        metadata_path,
    )

    local_path = Path(
        hf_hub_download(
            repo_id=cfg.dataset_id,
            repo_type="dataset",
            filename=metadata_path,
            revision=cfg.revision,
        )
    )

    yield from _iter_csv_rows(local_path)
def maybe_download_core_audio(
    *,
    enable: bool,
    dataset_id: str,
    out_audio_dir: Path,
    rows: list[dict],
    audio_key: str = "wav",
    revision: str | None = None,
    max_items: int = 0,
) -> None:
    """
    Optional helper used by the builder to export a small 'core' subset of audio.

    For now, this is intentionally a no-op unless enable=True and max_items>0.
    Full builds are metadata-only and should not require audio downloads.

    Once we confirm the exact HF schema for audio fields, we will implement:
    - downloading referenced wav paths
    - decoding HF Audio columns when present
    - writing standardized WAV files
    """
    if not enable or max_items <= 0:
        logger.info("Core audio export disabled (enable=%s, max_items=%s).", enable, max_items)
        return

    # Minimal placeholder behavior: raise with a clear message.
    raise NotImplementedError(
        "Core audio export is not enabled in this build yet. "
        "Run with --build full (metadata-only), or implement core export once HF audio schema is confirmed."
    )
