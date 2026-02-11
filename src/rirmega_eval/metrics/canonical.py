from __future__ import annotations

import numpy as np
from rirmega_eval.config import MetricConfig

from .center_time import center_time_s
from .clarity import clarity_db
from .drr import drr_db
from .rt60 import estimate_rt60_edt


def compute_all_metrics(x: np.ndarray, fs: int, cfg: MetricConfig | None = None) -> dict[str, float]:
    cfg = cfg or MetricConfig()
    rt60_s, edt_s, _ = estimate_rt60_edt(x, fs=fs, eps=cfg.eps)
    drr = drr_db(x, fs=fs, pre_ms=cfg.drr_pre_ms, direct_ms=cfg.drr_direct_ms, eps=cfg.eps)
    c50 = clarity_db(x, fs=fs, window_ms=cfg.c50_ms, eps=cfg.eps)
    c80 = clarity_db(x, fs=fs, window_ms=cfg.c80_ms, eps=cfg.eps)
    ts = center_time_s(x, fs=fs, eps=cfg.eps)
    return {
        "rt60_s": float(rt60_s),
        "edt_s": float(edt_s),
        "drr_db": float(drr),
        "c50_db": float(c50),
        "c80_db": float(c80),
        "ts_s": float(ts),
    }

