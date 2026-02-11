from __future__ import annotations

import numpy as np
from rirmega_eval.metrics.edc import schroeder_edc_db
from rirmega_eval.metrics.rt60 import estimate_rt60_edt


def test_edc_monotone(synthetic_exponential_rir):
    x, fs = synthetic_exponential_rir
    edc = schroeder_edc_db(x)
    # EDC in dB should be non-increasing (allow tiny numerical noise)
    dif = np.diff(edc)
    assert np.percentile(dif, 99) <= 1e-6


def test_rt60_positive(synthetic_exponential_rir):
    x, fs = synthetic_exponential_rir
    rt60, edt, method = estimate_rt60_edt(x, fs)
    assert np.isfinite(rt60)
    assert rt60 > 0.0
    assert method in ("t30", "t20", "none")

