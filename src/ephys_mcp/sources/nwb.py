"""Neurodata Without Borders (NWB) files, local or streamed.

Only a user-supplied local path is accepted here. Remote files are reached
through the DANDI source, which resolves URLs from the archive's own API.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .base import NeuralSource, SessionInfo, describe_trials

MAX_RAW_S = 10.0


def _behavior_series(nwb) -> dict[str, Any]:
    """Every numeric or boolean 1D/2D time series in processing modules or acquisition, keyed by name.

    Broadband ElectricalSeries are excluded; they are the neural signal, not behaviour.
    """
    from pynwb import TimeSeries
    from pynwb.ecephys import ElectricalSeries

    found: dict[str, Any] = {}

    def visit(obj, prefix: str):
        if isinstance(obj, ElectricalSeries):
            return
        if isinstance(obj, TimeSeries):
            data = obj.data
            if (
                data is not None
                and len(data.shape) in (1, 2)
                and (np.issubdtype(data.dtype, np.number) or data.dtype == bool)
            ):
                found[obj.name if obj.name not in found else f"{prefix}/{obj.name}"] = obj
            return
        for attr in ("spatial_series", "time_series"):
            for child in (getattr(obj, attr, None) or {}).values():
                visit(child, f"{prefix}/{obj.name}")

    for mod_name, mod in nwb.processing.items():
        for iface in mod.data_interfaces.values():
            visit(iface, mod_name)
    for obj in nwb.acquisition.values():
        visit(obj, "acquisition")
    return found


class NwbSource(NeuralSource):
    kind = "nwb"

    def __init__(self, path: str = "", *, _file=None, _uri: str = "", _license: str = "", _citation: str = ""):
        import h5py
        from pynwb import NWBHDF5IO
        from pynwb.ecephys import ElectricalSeries

        if _file is None:
            p = Path(path).expanduser()
            if p.suffix != ".nwb" or not p.is_file():
                raise ValueError(f"not a readable .nwb file: {path!r}")
            _file, _uri = p, p.resolve().as_uri()
        self._h5 = h5py.File(_file, "r")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cached-namespace version chatter
            self._io = NWBHDF5IO(file=self._h5, load_namespaces=True)
            self._nwb = self._io.read()
        nwb = self._nwb
        self._uri, self._license, self._citation = _uri, _license, _citation

        self._spikes: list[np.ndarray] = []
        self._valid: np.ndarray | None = None
        if nwb.units is not None and "spike_times" in nwb.units.colnames:
            flat = np.asarray(nwb.units["spike_times"].target.data[:], dtype=float)
            ends = np.asarray(nwb.units["spike_times"].data[:], dtype=int)
            self._spikes = [np.sort(a) for a in np.split(flat, ends[:-1])]
            if "obs_intervals" in nwb.units.colnames and len(nwb.units):
                self._valid = np.asarray(nwb.units["obs_intervals"][0], dtype=float).reshape(-1, 2)

        self._rate_fixes: set[str] = set()
        self._spike_span = (
            max(s[-1] for s in self._spikes if s.size) - min(s[0] for s in self._spikes if s.size)
            if any(s.size for s in self._spikes)
            else 0.0
        )

        self._positions, self._areas = self._electrode_geometry(nwb)

        self._trials: dict[str, np.ndarray] = {}
        if nwb.trials is not None:
            for name in nwb.trials.colnames:
                col = nwb.trials[name]
                if hasattr(col, "target"):  # ragged column
                    continue
                values = np.asarray(col.data[:])
                if values.ndim == 1 and values.dtype.kind in "biufUSO":
                    self._trials[name] = values.astype(str) if values.dtype.kind in "SO" else values

        self._behavior = _behavior_series(nwb)
        self._timestamps: dict[str, np.ndarray] = {}
        self._raw = next((a for a in nwb.acquisition.values() if isinstance(a, ElectricalSeries)), None)

        starts, stops = [], []
        if any(s.size for s in self._spikes):
            starts.append(min(s[0] for s in self._spikes if s.size))
            stops.append(max(s[-1] for s in self._spikes if s.size))
        if self._valid is not None:
            starts.append(self._valid[0, 0])
            stops.append(self._valid[-1, 1])
        if self._raw is not None:
            t = self._times(self._raw)
            starts.append(t[0])
            stops.append(t[-1])
        if not starts:
            raise ValueError("file has neither sorted units nor an ElectricalSeries; nothing to analyse")
        self._t0, self._t1 = float(min(starts)), float(max(stops))
        if self._t0 < 1.0:  # sessions that begin near zero are easier to reason about from zero
            self._t0 = 0.0

    def _electrode_geometry(self, nwb) -> tuple[np.ndarray | None, list[str] | None]:
        """Per-channel (or per-unit, when sorted) positions in um and area labels from the electrodes table."""
        table = nwb.electrodes
        if table is None or len(table) == 0:
            return None, None
        cols = table.colnames
        pos = None
        for xcol, ycol in (("rel_x", "rel_y"), ("x", "y")):
            if xcol in cols and ycol in cols:
                xy = np.stack([np.asarray(table[xcol][:], float), np.asarray(table[ycol][:], float)], axis=1)
                if np.isfinite(xy).all():
                    if np.abs(xy).max() < 0.05:  # NWB's x/y/z are nominally metres
                        xy = xy * 1e6
                    pos = xy
                    break
        areas = [str(a) for a in table["location"][:]] if "location" in cols else None
        if self._spikes and nwb.units is not None and "electrodes" in nwb.units.colnames:
            # one row per unit: take its first electrode
            rows = []
            for i in range(len(nwb.units)):
                idx = list(nwb.units["electrodes"][i].index)
                rows.append(idx[0] if idx else -1)
            rows = np.asarray(rows)
            ok = rows >= 0
            if pos is not None:
                pos = np.where(ok[:, None], pos[np.clip(rows, 0, len(pos) - 1)], np.nan)
                pos = None if not np.isfinite(pos).all() else pos
            if areas is not None:
                areas = [areas[r] if r >= 0 else "unknown" for r in rows]
        elif self._raw is not None and (pos is not None or areas is not None):
            n = self._raw.data.shape[1]
            if pos is not None and len(pos) != n:
                pos = None
            if areas is not None and len(areas) != n:
                areas = None
        return pos, areas

    def channel_positions(self) -> np.ndarray | None:
        return self._positions

    def channel_areas(self) -> list[str] | None:
        return self._areas

    def _stored_rate(self, series) -> float:
        """The series' sampling rate in Hz, correcting files that stored the period instead.

        Some published files put 0.02 (a 20 ms bin) in `rate`. When the rate taken literally
        implies a duration wildly longer than the spikes span but its reciprocal fits, use that.
        """
        rate = float(series.rate)
        n = series.data.shape[0]
        span = self._spike_span
        if span and rate < 1.0 and n / rate > 10 * span and abs(n * rate - span) < 0.5 * span:
            self._rate_fixes.add(series.name)
            return 1.0 / rate
        return rate

    def _times(self, series) -> np.ndarray:
        key = series.name
        if key not in self._timestamps:
            if series.timestamps is not None:
                self._timestamps[key] = np.asarray(series.timestamps[:], dtype=float)
            else:
                start = series.starting_time or 0.0
                self._timestamps[key] = start + np.arange(series.data.shape[0]) / self._stored_rate(series)
        return self._timestamps[key]

    def _rate(self, series) -> float | None:
        if series.rate:
            return round(self._stored_rate(series), 3)
        t = self._times(series)
        return round(float(1.0 / np.median(np.diff(t[:10_000]))), 3) if t.size > 1 else None

    def info(self) -> SessionInfo:
        nwb = self._nwb
        first = next(iter(self._behavior.values()), None)
        valid = self.valid_intervals()
        notes = [f"NWB session: {(nwb.session_description or '').strip()[:200]}"]
        if self._raw is None:
            notes.append("No broadband signal: sorted units only, so raw-signal tools are unavailable.")
        for k in self._behavior:
            self._times(self._behavior[k])  # so any rate fixes are known before the notes are written
        if self._rate_fixes:
            notes.append(
                f"The file stores a sampling period instead of a rate for {sorted(self._rate_fixes)}; "
                "the reciprocal was used."
            )
        return SessionInfo(
            source=self.kind,
            uri=self._uri,
            duration_s=round(self._t1 - self._t0, 3),
            n_channels=len(self._spikes) or (self._raw.data.shape[1] if self._raw is not None else 0),
            raw_fs_hz=self._rate(self._raw) if self._raw is not None else None,
            has_sorted_spikes=bool(self._spikes),
            behavior_signals={k: (v.data.shape[1] if len(v.data.shape) == 2 else 1) for k, v in self._behavior.items()},
            behavior_fs_hz=self._rate(first) if first is not None else None,
            license=self._license or "unknown (local file; check its origin)",
            citation=self._citation,
            notes=" ".join(notes),
            t_start_s=round(self._t0, 3),
            behavior_units={k: str(v.unit) for k, v in self._behavior.items()},
            recorded_fraction=round(float((valid[:, 1] - valid[:, 0]).sum() / (self._t1 - self._t0)), 3),
            has_probe_geometry=self._positions is not None,
            channel_areas={str(k): int(v) for k, v in zip(*np.unique(self._areas, return_counts=True))}
            if self._areas
            else {},
            **describe_trials(self._trials),
        )

    def valid_intervals(self) -> np.ndarray:
        if self._valid is not None:
            return self._valid
        return np.array([[self._t0, self._t1]])

    def trials(self) -> dict[str, np.ndarray]:
        return self._trials

    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        if not self._spikes:
            raise NotImplementedError("this file has no sorted units")
        return [s[np.searchsorted(s, t0) : np.searchsorted(s, t1)] for s in self._spikes]

    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._behavior:
            raise KeyError(f"unknown behavior signal {name!r}; available: {sorted(self._behavior)}")
        series = self._behavior[name]
        t = self._times(series)
        i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
        values = np.asarray(series.data[i0:i1], dtype=float)
        return t[i0:i1], values.reshape(len(values), -1)

    def read_raw(self, t0: float, t1: float, channels: list[int] | None = None) -> np.ndarray:
        if self._raw is None:
            raise NotImplementedError("this file has no broadband signal (sorted units only)")
        if t1 - t0 > MAX_RAW_S:
            raise ValueError(f"raw reads are limited to {MAX_RAW_S:g} s per call")
        t = self._times(self._raw)
        i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
        data = np.asarray(self._raw.data[i0:i1], dtype=np.float32)
        if channels is not None:
            data = data[:, channels]
        scale = float(self._raw.conversion or 1.0) * 1e6  # stored volts -> microvolts
        return data * scale

    def close(self) -> None:
        self._io.close()
        self._h5.close()
