"""Lab Streaming Layer: subscribe to a broadband stream on the local network.

Requires the `lsl` extra (`pip install ephys-mcp[lsl]`). Any acquisition
system that publishes an LSL outlet (OpenBCI, Intan and Blackrock bridges,
BrainFlow, custom rigs) can be read this way. Subscribing is read-only.
"""

from __future__ import annotations

import threading

import numpy as np

from .live import RingBufferSource

RESOLVE_TIMEOUT_S = 3.0
FORMATS = {1: "float32", 2: "float64", 3: "string", 4: "int32", 5: "int16", 6: "int8", 7: "int64"}


def _pylsl():
    try:
        import pylsl
    except ImportError as exc:
        raise NotImplementedError("LSL support needs the lsl extra: pip install 'ephys-mcp[lsl]'") from exc
    return pylsl


def list_streams(wait_s: float = RESOLVE_TIMEOUT_S) -> list[dict]:
    """Streams visible on the network right now."""
    pylsl = _pylsl()
    out = []
    for s in pylsl.resolve_streams(wait_time=wait_s):
        out.append(
            {
                "name": s.name(),
                "type": s.type(),
                "n_channels": s.channel_count(),
                "fs_hz": s.nominal_srate(),
                "format": FORMATS.get(int(s.channel_format()), "unknown"),
                "host": s.hostname(),
                "source_id": s.source_id(),
            }
        )
    return out


class LslSource(RingBufferSource):
    kind = "lsl"

    def __init__(self, name: str = "", type: str = "", buffer_s: float = 60.0, uv_per_count: float = 0.0):
        pylsl = _pylsl()
        if not name and not type:
            raise ValueError("give the stream's name or type; call list_lsl_streams to see what is available")
        prop, value = ("name", name) if name else ("type", type)
        found = pylsl.resolve_byprop(prop, value, timeout=RESOLVE_TIMEOUT_S)
        if not found:
            raise ValueError(f"no LSL stream with {prop}={value!r} found within {RESOLVE_TIMEOUT_S:g} s")
        stream = found[0]
        fs = stream.nominal_srate()
        if fs <= 0:
            raise ValueError("stream has an irregular sample rate; a fixed-rate broadband stream is required")
        if stream.channel_format() == pylsl.cf_string:
            raise ValueError("stream carries strings, not samples")
        super().__init__(
            stream.channel_count(),
            fs,
            buffer_s,
            amplitude_unit="uV" if uv_per_count > 0 else "ADC counts",
        )
        self._scale = float(uv_per_count) if uv_per_count > 0 else 1.0
        self._name, self._type, self._host = stream.name(), stream.type(), stream.hostname()
        self._inlet = pylsl.StreamInlet(stream, max_buflen=int(buffer_s) + 1, recover=True)
        self._inlet.open_stream(timeout=RESOLVE_TIMEOUT_S)  # connect now, so nothing sent after open is missed
        self._thread = threading.Thread(target=self._reader, name=f"lsl-{self._name}", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._closed.is_set():
            chunk, _ = self._inlet.pull_chunk(timeout=0.2, max_samples=4096)
            if chunk:
                self.push(np.asarray(chunk, dtype=np.float32) * self._scale)

    def uri(self) -> str:
        return f"lsl://{self._host}/{self._name}"

    def notes(self) -> str:
        return f"LSL stream {self._name!r} (type {self._type!r}) from {self._host}. " + super().notes()

    def close(self) -> None:
        super().close()
        self._thread.join(timeout=1.0)
        self._inlet.close_stream()
