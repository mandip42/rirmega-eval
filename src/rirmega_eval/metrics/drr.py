from __future__ import annotations

import numpy as np


def direct_index_max_abs(x: np.ndarray) -> int:
    return int(np.argmax(np.abs(x)))


def drr_db(
    x: np.ndarray,
    fs: int,
    pre_ms: float,
    direct_ms: float,
    eps: float = 1e-12,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    n0 = direct_index_max_abs(x)
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(direct_ms * 1e-3 * fs))
    a = max(0, n0 - pre)
    b = min(x.size, n0 + post)
    e = x * x
    e_dir = float(np.sum(e[a:b]))
    e_rev = float(np.sum(e[:a]) + np.sum(e[b:]))
    if e_dir <= 0.0 or e_rev <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((e_dir + eps) / (e_rev + eps)))

