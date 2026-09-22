"""Synthetic motor-cortex session with known ground truth.

Units are cosine-tuned to 2D cursor velocity and fire as inhomogeneous Poisson
processes. Broadband voltage is rendered on demand from the spike times, so
spike detection and decoding can both be validated against the truth.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .base import NeuralSource, SessionInfo, describe_trials

RAW_FS = 20_000.0
BEHAVIOR_FS = 100.0
NOISE_UV = 10.0
SPIKE_AMP_UV = -80.0
REFRACTORY_S = 0.002
SPREAD_UM = 40.0  # spatial decay of a unit's waveform across contacts on a dense probe
ONSET_SPEED = 15.0  # cm/s; an upward crossing marks a movement onset
DIRECTIONS = np.array(["right", "up", "left", "down"])


def _spike_template(fs: float) -> np.ndarray:
    t = np.arange(int(0.0015 * fs)) / fs
    w = np.exp(-(((t - 0.0004) / 0.00015) ** 2)) - 0.35 * np.exp(-(((t - 0.0009) / 0.0003) ** 2))
    return (w / np.abs(w).max()).astype(np.float32)


class SyntheticSource(NeuralSource):
    kind = "synthetic"

    def __init__(
        self,
        n_units: int = 32,
        duration_s: float = 120.0,
        noise: float = 1.0,
        seed: int = 0,
        pitch_um: float = 0.0,
    ):
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
            times = np.sort((idx + rng.uniform(0, 1, idx.size)) / BEHAVIOR_FS)
            keep = np.ones(times.size, dtype=bool)
            last = -np.inf
            for i, t in enumerate(times):  # absolute refractory period, as real neurons have
                if t - last < REFRACTORY_S:
                    keep[i] = False
                else:
                    last = t
            self._spikes.append(times[keep])
        self._template = _spike_template(RAW_FS)
        self._trials = self._find_movements()
        # With a pitch, channels sit on a linear probe and unit u lives at contact u, so its spikes
        # also appear, attenuated, on nearby contacts; without one, every channel is an isolated electrode.
        self.pitch_um = float(pitch_um)
        self._positions = np.c_[np.zeros(n_units), self.pitch_um * np.arange(n_units)] if pitch_um > 0 else None
        if self._positions is not None:
            d = np.abs(self._positions[:, 1][:, None] - self._positions[:, 1][None, :])
            self._mix = np.exp(-d / SPREAD_UM)  # [channel, unit] amplitude fraction
            self._mix[self._mix < 0.05] = 0.0
        else:
            self._mix = np.eye(n_units)

    def _find_movements(self) -> dict[str, np.ndarray]:
        speed = np.linalg.norm(self._vel, axis=1)
        fast = speed > ONSET_SPEED
        onsets = np.flatnonzero(fast[1:] & ~fast[:-1]) + 1
        after = int(0.2 * BEHAVIOR_FS)
        onsets = onsets[(onsets > BEHAVIOR_FS) & (onsets < len(speed) - BEHAVIOR_FS)]
        if onsets.size == 0:
            return {"move_onset_time": np.empty(0), "direction": np.empty(0, dtype=DIRECTIONS.dtype)}
        onsets = onsets[np.insert(np.diff(onsets) > 0.5 * BEHAVIOR_FS, 0, True)]
        mean_vel = np.array([self._vel[i : i + after].mean(axis=0) for i in onsets]).reshape(-1, 2)
        quadrant = np.round(np.arctan2(mean_vel[:, 1], mean_vel[:, 0]) / (np.pi / 2)).astype(int) % 4
        return {"move_onset_time": self._t[onsets], "direction": DIRECTIONS[quadrant]}

    def trials(self) -> dict[str, np.ndarray]:
        return self._trials

    def info(self) -> SessionInfo:
        return SessionInfo(
            source=self.kind,
            uri=f"synthetic://units={self.n_units}&duration={self.duration_s}&noise={self.noise}&seed={self.seed}"
            + (f"&pitch_um={self.pitch_um:g}" if self._positions is not None else ""),
            duration_s=self.duration_s,
            n_channels=self.n_units,
            raw_fs_hz=RAW_FS,
            has_sorted_spikes=True,
            behavior_signals={"cursor_velocity": 2, "cursor_position": 2},
            behavior_fs_hz=BEHAVIOR_FS,
            behavior_units={"cursor_velocity": "cm/s", "cursor_position": "cm"},
            license="CC0-1.0 (generated)",
            notes="Simulated data. One unit per channel, cosine-tuned to cursor velocity. "
            "Trials are detected movement onsets, grouped by reach direction."
            + (
                f" Channels form a linear probe at {self.pitch_um:g} um pitch, so units bleed onto neighbours."
                if self._positions is not None
                else ""
            ),
            has_probe_geometry=self._positions is not None,
            **describe_trials(self._trials),
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
        waves: dict[int, np.ndarray] = {}

        def wave(unit: int) -> np.ndarray:
            if unit not in waves:
                s = self._spikes[unit]
                idx = ((s[(s >= t0) & (s < t1)] - t0) * RAW_FS).astype(int)
                train = np.zeros(n, dtype=np.float32)
                train[idx[idx < n]] = SPIKE_AMP_UV
                waves[unit] = np.convolve(train, self._template, mode="same")
            return waves[unit]

        for j, ch in enumerate(channels):
            for unit in np.flatnonzero(self._mix[ch]):
                out[:, j] += self._mix[ch, unit] * wave(unit)
        return out

    def channel_positions(self) -> np.ndarray | None:
        return self._positions

    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        t0, t1 = self._clip(t0, t1)
        data = {"cursor_velocity": self._vel, "cursor_position": self._pos}.get(name)
        if data is None:
            raise KeyError(f"unknown behavior signal {name!r}")
        m = (self._t >= t0) & (self._t < t1)
        return self._t[m], data[m]
