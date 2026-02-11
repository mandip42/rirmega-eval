from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edc import schroeder_edc_db


@dataclass(frozen=True)
class DecayFit:
    rt60_s: float
    slope_db_per_s: float
    intercept_db: float
    method: str


def _fit_line(t: np.ndarray, y_db: np.ndarray) -> tuple[float, float]:
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, y_db, rcond=None)[0]
    return float(m), float(b)


def estimate_rt60_edt(x: np.ndarray, fs: int, eps: float = 1e-12) -> tuple[float, float, str]:
    """
    Canonical rules:
    - RT60: prefer T30 range [-5, -35] if available, else T20 [-5, -25], else NaN.
    - EDT: fit [0, -10].
    Returns: (rt60_s, edt_s, rt60_method)
    """
    edc = schroeder_edc_db(x, eps=eps)
    t = np.arange(edc.size) / float(fs)

    def fit_range(hi: float, lo: float) -> tuple[float, float] | None:
        # hi and lo are negative dB values (hi closer to 0)
        msk = (edc <= hi) & (edc >= lo)
        if np.count_nonzero(msk) < 8:
            return None
        return _fit_line(t[msk], edc[msk])

    # EDT
    edt_fit = fit_range(0.0, -10.0)
    if edt_fit is None:
        edt_s = float("nan")
    else:
        m, _ = edt_fit
        edt_s = float("nan") if m >= 0 else (-60.0 / m)

    # RT60
    fit = fit_range(-5.0, -35.0)
    method = "t30"
    if fit is None:
        fit = fit_range(-5.0, -25.0)
        method = "t20"
    if fit is None:
        return float("nan"), edt_s, "none"
    m, _ = fit
    rt60_s = float("nan") if m >= 0 else (-60.0 / m)
    return rt60_s, edt_s, method

