from __future__ import annotations

import numpy as np


def center_time_s(x: np.ndarray, fs: int, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    e = x * x
    den = float(np.sum(e))
    if den <= 0.0:
        return float("nan")
    t = np.arange(x.size, dtype=np.float64) / float(fs)
    return float(np.sum(t * e) / (den + eps))

