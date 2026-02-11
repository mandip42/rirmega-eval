from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEED: int = 1337

@dataclass(frozen=True)
class MetricConfig:
    eps: float = 1e-12
    # Direct arrival index selection:
    direct_pick: str = "max_abs"  # "max_abs" or "first_above"
    # DRR direct window
    drr_pre_ms: float = 1.0
    drr_direct_ms: float = 2.5
    # Clarity windows
    c50_ms: float = 50.0
    c80_ms: float = 80.0


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 1337
    ratios: tuple[float, float, float] = (0.82, 0.087, 0.093)


@dataclass(frozen=True)
class BuildConfig:
    build: str
    sources: dict[str, object]
    splits: SplitConfig
    metrics: MetricConfig
    max_core_rirs: int = 512
    max_core_pairs: int = 256

