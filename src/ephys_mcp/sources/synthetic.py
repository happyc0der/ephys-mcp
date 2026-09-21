"""Synthetic motor-cortex session with known ground truth.

Units are cosine-tuned to 2D cursor velocity and fire as inhomogeneous Poisson
processes. Broadband voltage is rendered on demand from the spike times, so
spike detection and decoding can both be validated against the truth.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .base import NeuralSource, SessionInfo

RAW_FS = 20_000.0
BEHAVIOR_FS = 100.0
NOISE_UV = 10.0
SPIKE_AMP_UV = -80.0


def _spike_template(fs: float) -> np.ndarray:
    t = np.arange(int(0.0015 * fs)) / fs
    w = np.exp(-(((t - 0.0004) / 0.00015) ** 2)) - 0.35 * np.exp(-(((t - 0.0009) / 0.0003) ** 2))
    return (w / np.abs(w).max()).astype(np.float32)


class SyntheticSource(NeuralSource):
    kind = "synthetic"

    def __init__(self, n_units: int = 32, duration_s: float = 120.0, noise: float = 1.0, seed: int = 0):
        if not 1 <= n_units <= 1024:
            raise ValueError("n_units must be in 1..1024")
        if not 1.0 <= duration_s <= 3600.0:
            raise ValueError("duration_s must be in 1..3600")
        self.n_units = n_units
        self.duration_s = float(duration_s)
        self.noise = float(noise)
        self.seed = seed
        rng = np.random.default_rng(seed)

        n = int(self.duration_s * BEHAVIOR_FS)
        self._t = np.arange(n) / BEHAVIOR_FS
        vel = gaussian_filter1d(rng.standard_normal((n, 2)), sigma=0.25 * BEHAVIOR_FS, axis=0)
        self._vel = 10.0 * vel / vel.std()  # cm/s
        self._pos = np.cumsum(self._vel, axis=0) / BEHAVIOR_FS

        theta = rng.uniform(0, 2 * np.pi, n_units)
        self.preferred_dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self.baseline_hz = rng.uniform(5, 20, n_units)
        self.gain = rng.uniform(0.5, 1.5, n_units)  # Hz per cm/s

        rate = self.baseline_hz + self.gain * (self._vel @ self.preferred_dirs.T)
        rate = np.clip(rate + self.noise * rng.standard_normal(rate.shape), 0.5, None)
        counts = rng.poisson(rate / BEHAVIOR_FS)
        self._spikes: list[np.ndarray] = []
        for u in range(n_units):
            idx = np.repeat(np.arange(n), counts[:, u])
            self._spikes.append(np.sort((idx + rng.uniform(0, 1, idx.size)) / BEHAVIOR_FS))
        self._template = _spike_template(RAW_FS)

    def info(self) -> SessionInfo:
        return SessionInfo(
            source=self.kind,
            uri=f"synthetic://units={self.n_units}&duration={self.duration_s}&noise={self.noise}&seed={self.seed}",
            duration_s=self.duration_s,
            n_channels=self.n_units,
            raw_fs_hz=RAW_FS,
            has_sorted_spikes=True,
            behavior_signals={"cursor_velocity": 2, "cursor_position": 2},
            behavior_fs_hz=BEHAVIOR_FS,
            license="CC0-1.0 (generated)",
            notes="Simulated data. One unit per channel, cosine-tuned to cursor velocity.",
        )

    def _clip(self, t0: float, t1: float) -> tuple[float, float]:
        t0, t1 = max(0.0, t0), min(self.duration_s, t1)
        if t1 <= t0:
            raise ValueError("empty time range")
        return t0, t1

    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        t0, t1 = self._clip(t0, t1)
        return [s[(s >= t0) & (s < t1)] for s in self._spikes]

    def read_raw(self, t0: float, t1: float, channels: list[int] | None = None) -> np.ndarray:
        t0, t1 = self._clip(t0, t1)
        if t1 - t0 > 10.0:
            raise ValueError("raw reads are limited to 10 s per call")
        channels = list(range(self.n_units)) if channels is None else channels
        n = round((t1 - t0) * RAW_FS)
        rng = np.random.default_rng((self.seed, int(t0 * 1000)))
        out = (NOISE_UV * self.noise * rng.standard_normal((n, len(channels)))).astype(np.float32)
        for j, ch in enumerate(channels):
            s = self._spikes[ch]
            idx = ((s[(s >= t0) & (s < t1)] - t0) * RAW_FS).astype(int)
            train = np.zeros(n, dtype=np.float32)
            train[idx[idx < n]] = SPIKE_AMP_UV
            out[:, j] += np.convolve(train, self._template, mode="same")
        return out

    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        t0, t1 = self._clip(t0, t1)
        data = {"cursor_velocity": self._vel, "cursor_position": self._pos}.get(name)
        if data is None:
            raise KeyError(f"unknown behavior signal {name!r}")
        m = (self._t >= t0) & (self._t < t1)
        return self._t[m], data[m]
