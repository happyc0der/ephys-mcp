"""Broadband recordings stored as WAV: a folder of mono clips, or one multi-channel file.

Only a user-supplied local path is accepted; nothing is downloaded. WAV samples
carry no physical unit, so amplitudes are ADC counts unless `uv_per_count` is given.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..processing.spikes import detect_spikes
from .base import NeuralSource, SessionInfo

MAX_FILES = 256
MAX_RAW_S = 10.0
MAX_DETECT_SAMPLES = 100_000_000  # across channels; beyond this, whole-session spike times are refused
CHUNK_S = 10.0


def _open(path: Path) -> tuple[float, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # harmless non-audio chunks
        try:
            fs, data = wavfile.read(path, mmap=True)
        except Exception as exc:  # scipy raises bare ValueError/others on malformed files
            raise ValueError(f"cannot read {path.name} as WAV: {exc}") from exc
    return float(fs), data if data.ndim == 2 else data[:, None]


class WavDirSource(NeuralSource):
    kind = "wav_dir"

    def __init__(self, path: str = "", max_files: int = 64, uv_per_count: float = 0.0):
        p = Path(path).expanduser()
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() == ".wav" and f.is_file())
        elif p.is_file() and p.suffix.lower() == ".wav":
            files = [p]
        else:
            raise ValueError(f"not a folder of .wav files or a .wav file: {path!r}")
        if not files:
            raise ValueError(f"no .wav files in {path!r}")
        self._n_available = len(files)
        files = files[: max(1, min(max_files, MAX_FILES))]

        opened = [_open(f) for f in files]
        rates = {fs for fs, _ in opened}
        if len(rates) != 1:
            raise ValueError(f"files have different sample rates: {sorted(rates)}")
        self._fs = rates.pop()
        self._simultaneous = len(files) == 1
        if not self._simultaneous and any(d.shape[1] != 1 for _, d in opened):
            raise ValueError("a folder must hold mono clips; open a multi-channel file on its own")
        self._data = [d for _, d in opened]  # memory-mapped, one (n, ch) array per file
        self._names = [f.name for f in files]
        self._n = min(d.shape[0] for d in self._data)  # clips are cut to the shortest so channels align
        self._n_channels = self._data[0].shape[1] if self._simultaneous else len(files)
        self._scale = float(uv_per_count) if uv_per_count > 0 else 1.0
        self._calibrated = uv_per_count > 0
        self._uri = p.resolve().as_uri()
        self._spikes: list[np.ndarray] | None = None

    def info(self) -> SessionInfo:
        notes = ["Broadband only: spike times are threshold crossings, not sorted units."]
        if not self._calibrated:
            notes.append("Amplitudes are raw ADC counts, not microvolts; pass uv_per_count to calibrate.")
        if not self._simultaneous:
            notes.append(
                f"{self._n_channels} of {self._n_available} clips opened, one per channel and cut to the shortest. "
                "Clips are separate recordings, so timing across channels is not meaningful. "
                f"Channels follow file-name order, {self._names[0]} to {self._names[-1]}."
            )
        return SessionInfo(
            source=self.kind,
            uri=self._uri,
            duration_s=round(self._n / self._fs, 3),
            n_channels=self._n_channels,
            raw_fs_hz=self._fs,
            has_sorted_spikes=False,
            amplitude_unit="uV" if self._calibrated else "ADC counts",
            license="unknown (local files; check their origin before sharing results)",
            notes=" ".join(notes),
        )

    def _read(self, i0: int, i1: int, channels: list[int]) -> np.ndarray:
        if self._simultaneous:
            out = np.asarray(self._data[0][i0:i1, channels], dtype=np.float32)
        else:
            out = np.stack([np.asarray(self._data[c][i0:i1, 0], dtype=np.float32) for c in channels], axis=1)
        return out * self._scale

    def read_raw(self, t0: float, t1: float, channels: list[int] | None = None) -> np.ndarray:
        if t1 - t0 > MAX_RAW_S:
            raise ValueError(f"raw reads are limited to {MAX_RAW_S:g} s per call")
        channels = list(range(self._n_channels)) if channels is None else channels
        if any(not 0 <= c < self._n_channels for c in channels):
            raise ValueError(f"channels must be in 0..{self._n_channels - 1}")
        i0, i1 = max(0, round(t0 * self._fs)), min(self._n, round(t1 * self._fs))
        if i1 <= i0:
            raise ValueError("empty time range")
        return self._read(i0, i1, channels)

    def _detect_all(self) -> list[np.ndarray]:
        if self._n * self._n_channels > MAX_DETECT_SAMPLES:
            raise NotImplementedError(
                "recording too large for whole-session spike detection; use detect_spikes on a window instead"
            )
        chunk = int(CHUNK_S * self._fs)
        parts: list[list[np.ndarray]] = [[] for _ in range(self._n_channels)]
        for i0 in range(0, self._n, chunk):
            i1 = min(self._n, i0 + chunk)
            if i1 - i0 < 64:  # too short to filter
                break
            spikes, _ = detect_spikes(self._read(i0, i1, list(range(self._n_channels))), self._fs)
            for c, s in enumerate(spikes):
                parts[c].append(s + i0 / self._fs)
        return [np.concatenate(p) if p else np.empty(0) for p in parts]

    def spike_times(self, t0: float, t1: float) -> list[np.ndarray]:
        if self._spikes is None:
            self._spikes = self._detect_all()
        return [s[np.searchsorted(s, t0) : np.searchsorted(s, t1)] for s in self._spikes]

    def behavior(self, name: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        raise KeyError(f"unknown behavior signal {name!r}; WAV recordings carry no behaviour")

    def close(self) -> None:
        self._data.clear()
