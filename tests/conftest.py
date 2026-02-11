from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def synthetic_exponential_rir():
    # Exponential decay RIR with known-ish decay time.
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    tau = 0.25  # seconds
    x = np.exp(-t / tau)
    x[0] = 1.0
    return x.astype(np.float64), fs

