from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class FeatureConfig:
    max_seconds: float = 2.0
    stft_n_fft: int = 1024
    stft_hop: int = 256


def _crop(x: np.ndarray, fs: int, max_seconds: float) -> np.ndarray:
    n = int(max_seconds * fs)
    if x.shape[0] <= n:
        return x
    return x[:n]


def extract_features(x: np.ndarray, fs: int, cfg: FeatureConfig) -> dict[str, float]:
    """
    Cheap, stable features for a small baseline:
    - log energy stats
    - envelope decay proxy
    - spectral centroid / bandwidth proxies (from STFT magnitude)
    """
    x = np.asarray(x, dtype=np.float64)
    x = _crop(x, fs, cfg.max_seconds)
    eps = 1e-12

    e = x * x
    loge = np.log(e + eps)
    feats: dict[str, float] = {}
    feats["loge_mean"] = float(np.mean(loge))
    feats["loge_std"] = float(np.std(loge))
    feats["loge_p95"] = float(np.quantile(loge, 0.95))
    feats["loge_p05"] = float(np.quantile(loge, 0.05))

    # Simple decay proxy: slope of log-energy envelope over time
    win = max(64, int(0.002 * fs))
    env = np.convolve(e, np.ones(win) / win, mode="same") + eps
    t = np.arange(env.size) / fs
    y = 10.0 * np.log10(env)
    # Fit a line over the region after the peak to avoid direct spike dominance
    i0 = int(np.argmax(env))
    if i0 < y.size - 10:
        tt = t[i0:]
        yy = y[i0:]
        A = np.vstack([tt, np.ones_like(tt)]).T
        m, b = np.linalg.lstsq(A, yy, rcond=None)[0]
        feats["decay_slope_db_per_s"] = float(m)
        feats["decay_intercept_db"] = float(b)
    else:
        feats["decay_slope_db_per_s"] = 0.0
        feats["decay_intercept_db"] = float(y[i0])

    # STFT magnitude summary
    f, _, Zxx = stft(x, fs=fs, nperseg=cfg.stft_n_fft, noverlap=cfg.stft_n_fft - cfg.stft_hop)
    mag = np.abs(Zxx) + eps
    mag_mean = np.mean(mag, axis=1)
    w = mag_mean / (np.sum(mag_mean) + eps)
    feats["spec_centroid_hz"] = float(np.sum(f * w))
    feats["spec_bandwidth_hz"] = float(np.sqrt(np.sum(((f - feats["spec_centroid_hz"]) ** 2) * w)))

    return feats


def vectorize(feats: dict[str, float], keys: tuple[str, ...]) -> np.ndarray:
    return np.array([feats[k] for k in keys], dtype=np.float32)

