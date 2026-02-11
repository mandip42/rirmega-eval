from __future__ import annotations

import numpy as np
from rirmega_eval.metrics.center_time import center_time_s
from rirmega_eval.metrics.clarity import clarity_db


def test_clarity_defined():
    fs = 48000
    x = np.zeros(fs, dtype=np.float64)
    x[0] = 1.0
    x[int(0.1 * fs)] = 0.1
    c50 = clarity_db(x, fs, window_ms=50.0)
    c80 = clarity_db(x, fs, window_ms=80.0)
    assert np.isfinite(c50)
    assert np.isfinite(c80)


def test_center_time():
    fs = 48000
    x = np.zeros(fs, dtype=np.float64)
    x[int(0.2 * fs)] = 1.0
    ts = center_time_s(x, fs)
    assert abs(ts - 0.2) < 1e-3

