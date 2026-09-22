"""Spike sorting of a broadband window with spikeinterface's built-in sorters.

Needs the `sort` extra. Channels are treated as independent electrodes
because sources carry no probe geometry yet; units are therefore found per
channel, which suits single-electrode arrays but not dense probes.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings

import numpy as np

from ..sources.base import NeuralSource, SessionInfo

SORTERS = ("spykingcircus2", "tridesclous2")
MAX_SAMPLES = 60_000_000  # samples x channels held in memory for one sort
CHANNEL_PITCH_UM = 1000.0  # far enough apart that sorters never merge across channels
ISI_VIOLATION_S = 0.0015


def _spikeinterface():
    try:
        import spikeinterface as si
        import spikeinterface.sorters as ss
        from probeinterface import Probe
    except ImportError as exc:
        raise NotImplementedError("spike sorting needs the sort extra: pip install 'ephys-mcp[sort]'") from exc
    return si, ss, Probe


def sort_window(
    src: NeuralSource, t0: float, t1: float, sorter: str, channels: list[int] | None, chunk_s: float = 10.0
) -> tuple[list[np.ndarray], list[dict]]:
    """Sort [t0, t1) and return (spike trains per unit, per-unit summaries)."""
    si, ss, Probe = _spikeinterface()
    if sorter not in SORTERS:
        raise ValueError(f"sorter must be one of {SORTERS}")
    fs = src.info().raw_fs_hz
    if fs is None:
        raise ValueError("this session has no broadband signal to sort")
    n_ch = len(channels) if channels else src.info().n_channels
    if (t1 - t0) * fs * n_ch > MAX_SAMPLES:
        raise ValueError(f"window too large: keep duration x channels under {MAX_SAMPLES / fs:.0f} channel-seconds")

    parts, start = [], t0
    while start < t1 - 1e-9:
        end = min(t1, start + chunk_s)
        parts.append(src.read_raw(start, end, channels))
        start = end
    raw = np.concatenate(parts)

    rec = si.NumpyRecording(raw, sampling_frequency=fs)
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=np.c_[np.zeros(n_ch), CHANNEL_PITCH_UM * np.arange(n_ch)], shapes="circle", shape_params={"radius": 5}
    )
    probe.set_device_channel_indices(np.arange(n_ch))
    rec = rec.set_probe(probe) or rec

    with tempfile.TemporaryDirectory(prefix="ephys-sort-") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            rec = rec.save(folder=f"{tmp}/rec", verbose=False, progress_bar=False)
            sorting = ss.run_sorter(
                sorter,
                rec,
                folder=f"{tmp}/out",
                verbose=False,
                remove_existing_folder=True,
                job_kwargs={"n_jobs": 1, "progress_bar": False},
            )
        trains, units = [], []
        for uid in sorting.unit_ids:
            idx = np.asarray(sorting.get_unit_spike_train(uid), dtype=int)
            idx = idx[(idx >= 0) & (idx < len(raw))]
            if idx.size == 0:
                continue
            win = int(0.001 * fs)
            peaks = np.array([raw[max(0, i - win) : i + win].min(axis=0) for i in idx[:500]])
            best = int(np.argmin(np.median(peaks, axis=0)))
            isi = np.diff(idx) / fs
            trains.append(t0 + idx / fs)
            units.append(
                {
                    "unit": len(units),
                    "channel": channels[best] if channels else best,
                    "n_spikes": int(idx.size),
                    "rate_hz": round(float(idx.size / (t1 - t0)), 2),
                    "peak_amplitude": round(float(np.median(peaks[:, best])), 1),
                    "isi_violation_fraction": round(float((isi < ISI_VIOLATION_S).mean()) if isi.size else 0.0, 4),
                }
            )
    return trains, units


class SortedSource(NeuralSource):
    """A session whose spike times come from a sorter over one window, everything else delegated."""

    def __init__(self, inner: NeuralSource, trains: list[np.ndarray], t0: float, t1: float, sorter: str):
        self.inner, self.trains, self.t0, self.t1, self.sorter = inner, trains, t0, t1, sorter
        self.kind = inner.kind

    def info(self) -> SessionInfo:
        info = self.inner.info()
        info.has_sorted_spikes = True
        info.n_channels = len(self.trains)
        info.notes = (
            f"Spike times are {len(self.trains)} units sorted by {self.sorter} over {self.t0:g}-{self.t1:g} s; "
            f"analyses are limited to that window. Original: {info.notes}"
        )
        return info

    def valid_intervals(self) -> np.ndarray:
        iv = np.clip(self.inner.valid_intervals(), self.t0, self.t1)
        return iv[iv[:, 1] > iv[:, 0]]

    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        return [s[np.searchsorted(s, t0) : np.searchsorted(s, t1)] for s in self.trains]

    def read_raw(self, t0, t1, channels=None):
        return self.inner.read_raw(t0, t1, channels)

    def behavior(self, name, t0, t1):
        return self.inner.behavior(name, t0, t1)

    def trials(self):
        return self.inner.trials()

    def close(self) -> None:
        self.inner.close()
