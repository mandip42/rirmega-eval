from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def schroeder_edc_db(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    x64 = np.asarray(x, dtype=np.float64)
    e = x64 * x64

    tail = np.cumsum(e[::-1])[::-1]
    tail = tail / (tail[0] + eps)

    edc_db = 10.0 * np.log10(tail + eps)
    return np.asarray(edc_db, dtype=np.float64)
