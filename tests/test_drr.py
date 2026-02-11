from __future__ import annotations

import numpy as np
from rirmega_eval.metrics.drr import drr_db


def test_drr_higher_for_more_direct_energy():
    fs = 48000
    x1 = np.zeros(fs, dtype=np.float64)
    x1[0] = 1.0
    x1[1000:] = 0.01

    x2 = np.zeros(fs, dtype=np.float64)
    x2[0] = 0.5
    x2[1000:] = 0.02

    d1 = drr_db(x1, fs, pre_ms=1.0, direct_ms=2.5)
    d2 = drr_db(x2, fs, pre_ms=1.0, direct_ms=2.5)
    assert d1 > d2

