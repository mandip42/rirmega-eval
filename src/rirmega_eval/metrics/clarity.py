from __future__ import annotations

import numpy as np

from .drr import direct_index_max_abs


def clarity_db(x: np.ndarray, fs: int, window_ms: float, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    n0 = direct_index_max_abs(x)
    w = int(round(window_ms * 1e-3 * fs))
    e = x * x
    early = float(np.sum(e[n0 : min(x.size, n0 + w)]))
    late = float(np.sum(e[min(x.size, n0 + w) :]))
    if early <= 0.0 or late <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((early + eps) / (late + eps)))

