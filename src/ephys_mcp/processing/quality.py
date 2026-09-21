"""Per-channel signal quality from a short broadband snippet."""

from __future__ import annotations

import numpy as np

from .spikes import bandpass, detect_spikes


def signal_quality(raw: np.ndarray, fs: float) -> dict:
    spikes, sigma = detect_spikes(raw, fs)
    filt = bandpass(raw, fs)
    dur = raw.shape[0] / fs
    snr = np.zeros(raw.shape[1])
    for ch, s in enumerate(spikes):
        if s.size:
            idx = np.clip((s * fs).astype(int), 0, raw.shape[0] - 1)
            half = int(0.0005 * fs)
            peaks = [np.abs(filt[max(0, i - half) : i + half, ch]).max() for i in idx]
            snr[ch] = float(np.mean(peaks) / sigma[ch])
    rates = np.array([s.size / dur for s in spikes])
    med = float(np.median(sigma))
    return {
        "snippet_s": round(dur, 3),
        "median_noise_uv": round(med, 2),
        "median_snr": round(float(np.median(snr)), 2),
        "median_threshold_rate_hz": round(float(np.median(rates)), 2),
        "dead_channels": [int(i) for i in np.flatnonzero(sigma < 0.2 * med)],
        "noisy_channels": [int(i) for i in np.flatnonzero(sigma > 3.0 * med)],
        "low_snr_channels": [int(i) for i in np.flatnonzero(snr < 3.0)],
    }
