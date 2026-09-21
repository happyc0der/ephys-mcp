"""Threshold-crossing spike detection on broadband voltage."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(raw: np.ndarray, fs: float, lo: float = 300.0, hi: float = 3000.0) -> np.ndarray:
    sos = butter(3, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, raw, axis=0)


def robust_sigma(x: np.ndarray) -> np.ndarray:
    """Noise estimate per channel from the median absolute deviation."""
    return np.median(np.abs(x), axis=0) / 0.6745


def detect_spikes(
    raw: np.ndarray, fs: float, threshold_sigma: float = -4.5, refractory_s: float = 0.001
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (spike times in seconds per channel, noise sigma per channel)."""
    filt = bandpass(raw, fs)
    sigma = robust_sigma(filt)
    refractory = max(1, int(refractory_s * fs))
    out = []
    for ch in range(filt.shape[1]):
        x = filt[:, ch]
        thr = threshold_sigma * sigma[ch]
        below = x < thr if threshold_sigma < 0 else x > thr
        onsets = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        keep, last = [], -refractory
        for i in onsets:
            if i - last >= refractory:
                keep.append(i)
                last = i
        out.append(np.asarray(keep, dtype=float) / fs)
    return out, sigma


def match_spikes(detected: np.ndarray, truth: np.ndarray, tol_s: float = 0.001) -> dict:
    """Precision/recall of detected spike times against ground truth."""
    if truth.size == 0 or detected.size == 0:
        return {"precision": 0.0, "recall": 0.0}
    idx = np.clip(np.searchsorted(truth, detected), 1, truth.size - 1)
    nearest = np.minimum(np.abs(detected - truth[idx - 1]), np.abs(detected - truth[idx]))
    tp = int((nearest <= tol_s).sum())
    return {"precision": tp / detected.size, "recall": min(1.0, tp / truth.size)}
