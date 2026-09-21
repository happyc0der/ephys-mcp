"""Peri-stimulus time histograms: firing aligned to an event, averaged over trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Psth:
    t: np.ndarray  # bin centres relative to the event, seconds
    rate: np.ndarray  # (n_units, n_bins) trial-averaged rate in Hz
    sem: np.ndarray  # (n_units, n_bins) standard error over trials
    n_events: int


def usable_events(events: np.ndarray, window: tuple[float, float], valid: np.ndarray) -> np.ndarray:
    """Mask of events whose whole window lies inside one recorded span."""
    events = np.asarray(events, dtype=float)
    lo, hi = events + window[0], events + window[1]
    k = np.clip(np.searchsorted(valid[:, 0], lo, side="right") - 1, 0, len(valid) - 1)
    return np.isfinite(events) & (lo >= valid[k, 0]) & (hi <= valid[k, 1])


def aligned_counts(spikes: list[np.ndarray], events: np.ndarray, window: tuple[float, float], bin_s: float):
    """(bin centres, counts[n_events, n_units, n_bins])."""
    n_bins = round((window[1] - window[0]) / bin_s)
    edges = window[0] + bin_s * np.arange(n_bins + 1)
    counts = np.zeros((len(events), len(spikes), n_bins))
    for u, s in enumerate(spikes):
        for e, t in enumerate(events):
            seg = s[np.searchsorted(s, t + edges[0]) : np.searchsorted(s, t + edges[-1])]
            counts[e, u] = np.histogram(seg - t, edges)[0]
    return edges[:-1] + bin_s / 2, counts


def compute_psth(spikes: list[np.ndarray], events: np.ndarray, window: tuple[float, float], bin_s: float) -> Psth:
    t, counts = aligned_counts(spikes, events, window, bin_s)
    rate = counts / bin_s
    n = max(len(events), 1)
    sem = rate.std(axis=0, ddof=1) / np.sqrt(n) if len(events) > 1 else np.zeros(rate.shape[1:])
    return Psth(t, rate.mean(axis=0) if len(events) else np.zeros(rate.shape[1:]), sem, len(events))


def modulation(psth: Psth) -> np.ndarray:
    """Per unit: largest post-event departure from the pre-event baseline, in baseline SDs."""
    pre = psth.t < 0
    if not pre.any() or pre.all():
        return np.zeros(len(psth.rate))
    base = psth.rate[:, pre].mean(axis=1, keepdims=True)
    sd = psth.rate[:, pre].std(axis=1, keepdims=True) + 1.0  # +1 Hz keeps near-silent units from dominating
    z = (psth.rate[:, ~pre] - base) / sd
    return z[np.arange(len(z)), np.abs(z).argmax(axis=1)]


def summarize(psth: Psth, max_points: int = 25, top: int = 5) -> dict:
    pop = psth.rate.mean(axis=0)
    pre = psth.t < 0
    z = modulation(psth)
    order = np.argsort(-np.abs(z))[:top]
    step = max(1, len(psth.t) // max(max_points, 1))
    return {
        "n_events": psth.n_events,
        "baseline_hz": round(float(pop[pre].mean()), 2) if pre.any() else None,
        "peak_hz": round(float(pop.max()), 2),
        "peak_time_s": round(float(psth.t[pop.argmax()]), 3),
        "population_psth": [[round(float(a), 3), round(float(b), 2)] for a, b in zip(psth.t[::step], pop[::step])],
        "most_modulated_units": [{"unit": int(u), "z": round(float(z[u]), 1)} for u in order],
    }
