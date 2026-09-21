"""Spike binning and firing-rate summaries."""

from __future__ import annotations

import numpy as np


def bin_spikes(spikes: list[np.ndarray], t0: float, t1: float, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin centres, counts[n_bins, n_units])."""
    n_bins = int(np.floor((t1 - t0) / bin_s))
    edges = t0 + bin_s * np.arange(n_bins + 1)
    counts = np.stack([np.histogram(s, edges)[0] for s in spikes], axis=1)
    return edges[:-1] + bin_s / 2, counts.astype(float)


def firing_stats(spikes: list[np.ndarray], t0: float, t1: float) -> dict:
    dur = t1 - t0
    rates = np.array([s.size / dur for s in spikes])
    order = np.argsort(rates)[::-1]
    return {
        "n_units": len(spikes),
        "mean_rate_hz": round(float(rates.mean()), 2),
        "median_rate_hz": round(float(np.median(rates)), 2),
        "min_rate_hz": round(float(rates.min()), 2),
        "max_rate_hz": round(float(rates.max()), 2),
        "silent_units": [int(i) for i in np.flatnonzero(rates < 0.1)],
        "most_active": [{"unit": int(i), "rate_hz": round(float(rates[i]), 2)} for i in order[:5]],
    }


def resample_to(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_dst, t_src, values[:, d]) for d in range(values.shape[1])], axis=1)
