"""The contract every neural data source implements.

A source is read-only by design: there is no write, stimulate or configure
method, and adapters must not add one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np

MAX_GROUPS = 8  # one validated categorical colour each


def describe_trials(trials: dict[str, np.ndarray]) -> dict:
    """The SessionInfo fields that summarise a trial table."""
    n = len(next(iter(trials.values()), []))
    events = [k for k in trials if k.endswith("_time")]
    groups = [k for k, v in trials.items() if k not in events and 1 < len(np.unique(v)) <= MAX_GROUPS]
    return {"n_trials": n, "event_columns": events, "group_columns": groups}


@dataclass
class SessionInfo:
    source: str
    uri: str
    duration_s: float
    n_channels: int
    raw_fs_hz: float | None  # None when the source has no broadband signal
    has_sorted_spikes: bool
    behavior_signals: dict[str, int] = field(default_factory=dict)  # name -> n dims
    behavior_fs_hz: float | None = None
    license: str = "unknown"
    citation: str = ""
    notes: str = ""
    t_start_s: float = 0.0  # session times run from t_start_s to t_start_s + duration_s
    behavior_units: dict[str, str] = field(default_factory=dict)
    recorded_fraction: float = 1.0  # share of the session covered by valid_intervals
    amplitude_unit: str = "uV"  # unit of read_raw values; uncalibrated sources say so
    has_probe_geometry: bool = False
    channel_areas: dict[str, int] = field(default_factory=dict)  # brain area label -> number of channels/units
    n_trials: int = 0
    event_columns: list[str] = field(default_factory=list)  # trial columns holding event times
    group_columns: list[str] = field(default_factory=list)  # trial columns usable to group trials

    def to_dict(self) -> dict:
        return asdict(self)


class NeuralSource(ABC):
    """Read-only access to one recording session."""

    kind: str = "abstract"

    @abstractmethod
    def info(self) -> SessionInfo: ...

    @abstractmethod
    def read_raw(self, t0: float, t1: float, channels: list[int] | None = None) -> np.ndarray:
        """Broadband voltage in microvolts, shape (n_samples, n_channels).

        Raises NotImplementedError when the source has no broadband signal.
        """

    @abstractmethod
    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        """Spike times in seconds, one array per channel/unit, within [t0, t1)."""

    @abstractmethod
    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps, values) for a behavioural signal; values is (n, dims)."""

    def valid_intervals(self) -> np.ndarray:
        """(n, 2) array of [start, stop) spans where data was actually recorded.

        Many datasets only record during trials. Analyses must ignore time outside
        these spans rather than treat it as silence.
        """
        info = self.info()
        return np.array([[info.t_start_s, info.t_start_s + info.duration_s]])

    def channel_positions(self) -> np.ndarray | None:
        """(n_channels, 2) contact positions in micrometres, or None when unknown."""
        return None

    def channel_areas(self) -> list[str] | None:
        """Brain area label per channel/unit, or None when unknown."""
        return None

    def trials(self) -> dict[str, np.ndarray]:
        """Trial table as equal-length 1D columns. Columns of event times end in `_time`."""
        return {}

    def close(self) -> None:
        pass
