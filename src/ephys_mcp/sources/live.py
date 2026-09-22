"""Base for live sources: a device pushes broadband samples into a ring buffer.

Session time is seconds since the source was opened. Only the most recent
`buffer_s` of signal is kept, so `valid_intervals()` slides forward and reads
outside it fail with a clear message. Spike times are threshold crossings
computed from the buffer on demand.

A device adapter subclasses this and calls `push(samples)` from its own reader
thread. Nothing here can write to a device.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..processing.spikes import detect_spikes
from .base import NeuralSource, SessionInfo

MAX_RAW_S = 10.0
MAX_BUFFER_S = 600.0


class RingBufferSource(NeuralSource):
    kind = "live"

    def __init__(self, n_channels: int, fs: float, buffer_s: float = 60.0, amplitude_unit: str = "ADC counts"):
        if not 1.0 <= buffer_s <= MAX_BUFFER_S:
            raise ValueError(f"buffer_s must be in 1..{MAX_BUFFER_S:g}")
        self.n_channels, self.fs, self.buffer_s = int(n_channels), float(fs), float(buffer_s)
        self.amplitude_unit = amplitude_unit
        self._buf = np.zeros((int(buffer_s * fs), self.n_channels), dtype=np.float32)
        self._written = 0  # total samples ever pushed; sample i lives at i % len(buf) while unevicted
        self._dropped = 0
        self._lock = threading.Lock()
        self._opened_at = time.monotonic()
        self._last_push_at: float | None = None
        self._closed = threading.Event()

    # ---- device side -------------------------------------------------------

    def push(self, samples: np.ndarray) -> None:
        """Append (n, n_channels) samples. Called from the adapter's reader thread."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, self.n_channels)
        n, size = len(samples), len(self._buf)
        if n == 0:
            return
        with self._lock:
            first = self._written  # global index of samples[0]
            if n > size:  # only the newest `size` samples can survive
                self._dropped += n - size
                first += n - size
                samples = samples[-size:]
                self._written += n - size
                n = size
            start = first % size
            end = start + n
            if end <= size:
                self._buf[start:end] = samples
            else:
                k = size - start
                self._buf[start:] = samples[:k]
                self._buf[: n - k] = samples[k:]
            self._written += n
            self._last_push_at = time.monotonic()

    # ---- contract ----------------------------------------------------------

    def _span(self) -> tuple[float, float]:
        """[oldest, newest) sample times currently in the buffer, in session seconds."""
        with self._lock:
            newest = self._written
        oldest = max(0, newest - len(self._buf))
        return oldest / self.fs, newest / self.fs

    def valid_intervals(self) -> np.ndarray:
        return np.array([list(self._span())])

    def status(self) -> dict:
        t0, t1 = self._span()
        return {
            "buffered_s": round(t1 - t0, 2),
            "buffer_capacity_s": self.buffer_s,
            "newest_sample_s": round(t1, 3),
            "samples_received": int(self._written),
            "samples_dropped": int(self._dropped),
            "seconds_since_last_data": None
            if self._last_push_at is None
            else round(time.monotonic() - self._last_push_at, 2),
            "receiving": self._last_push_at is not None and time.monotonic() - self._last_push_at < 2.0,
        }

    def info(self) -> SessionInfo:
        t0, t1 = self._span()
        return SessionInfo(
            source=self.kind,
            uri=self.uri(),
            duration_s=round(t1 - t0, 3),
            n_channels=self.n_channels,
            raw_fs_hz=self.fs,
            has_sorted_spikes=False,
            amplitude_unit=self.amplitude_unit,
            license="n/a (live stream)",
            notes=self.notes(),
            t_start_s=round(t0, 3),
        )

    def uri(self) -> str:
        return "live://"

    def notes(self) -> str:
        return (
            "Live stream. Session time is seconds since open; only the most recent buffer is readable and "
            "duration_s and t_start_s move forward. Spike times are threshold crossings, not sorted units."
        )

    def read_raw(self, t0: float, t1: float, channels: list[int] | None = None) -> np.ndarray:
        if t1 - t0 > MAX_RAW_S:
            raise ValueError(f"raw reads are limited to {MAX_RAW_S:g} s per call")
        lo, hi = self._span()
        if t0 < lo - 1e-9 or t1 > hi + 1e-9:
            raise ValueError(f"only {lo:.2f}..{hi:.2f} s is buffered; ask for a window inside it")
        i0, i1 = round(t0 * self.fs), round(t1 * self.fs)
        if i1 <= i0:
            raise ValueError("empty time range")
        channels = list(range(self.n_channels)) if channels is None else channels
        if any(not 0 <= c < self.n_channels for c in channels):
            raise ValueError(f"channels must be in 0..{self.n_channels - 1}")
        size = len(self._buf)
        idx = np.arange(i0, i1) % size
        with self._lock:
            if i0 < self._written - size:  # evicted between the check and the copy
                raise ValueError("that window has already left the buffer")
            return self._buf[idx][:, channels].copy()

    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        lo, hi = self._span()
        t0, t1 = max(t0, lo), min(t1, hi)
        if t1 <= t0:
            raise ValueError(f"only {lo:.2f}..{hi:.2f} s is buffered")
        chunk = MAX_RAW_S
        parts: list[list[np.ndarray]] = [[] for _ in range(self.n_channels)]
        start = t0
        while start < t1 - 64 / self.fs:
            end = min(t1, start + chunk)
            spikes, _ = detect_spikes(self.read_raw(start, end), self.fs)
            for c, s in enumerate(spikes):
                parts[c].append(s + start)
            start = end
        return [np.concatenate(p) if p else np.empty(0) for p in parts]

    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        raise KeyError(f"unknown behavior signal {name!r}; this live stream carries no behaviour")

    def close(self) -> None:
        self._closed.set()
