import threading
import time

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.spikes import match_spikes
from ephys_mcp.sources import SyntheticSource
from ephys_mcp.sources.live import RingBufferSource

FS = 20_000


class FakeDevice(RingBufferSource):
    """What a real adapter looks like: a thread that pushes samples from somewhere."""

    kind = "fake_device"

    def __init__(self, raw: np.ndarray, buffer_s: float, chunk: int = 2_000):
        super().__init__(raw.shape[1], FS, buffer_s)
        self._raw, self._chunk = raw, chunk
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        for i in range(0, len(self._raw), self._chunk):
            if self._closed.is_set():
                return
            self.push(self._raw[i : i + self._chunk])
            time.sleep(0.002)


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_ring_buffer_serves_recent_signal_and_evicts_old():
    sim = SyntheticSource(n_units=4, duration_s=6.0, seed=7)
    raw = sim.read_raw(0, 6.0)
    src = FakeDevice(raw, buffer_s=2.0)
    wait_until(lambda: src.status()["newest_sample_s"] >= 6.0)
    info = src.info()
    assert info.t_start_s == pytest.approx(4.0) and info.duration_s == pytest.approx(2.0)
    assert src.valid_intervals().tolist() == [[4.0, 6.0]]
    np.testing.assert_array_equal(src.read_raw(4.5, 5.0), raw[90_000:100_000])
    with pytest.raises(ValueError, match="buffered"):
        src.read_raw(1.0, 1.5)
    detected = src.spike_times(4.0, 6.0)
    truth = sim.spike_times(4.0, 6.0)
    assert np.median([match_spikes(d, t)["recall"] for d, t in zip(detected, truth)]) > 0.85
    src.close()


def test_oversized_push_keeps_the_newest_samples():
    src = RingBufferSource(1, 100.0, buffer_s=1.0)
    src.push(np.arange(250, dtype=np.float32))
    assert src.status()["samples_dropped"] == 150
    np.testing.assert_array_equal(src.read_raw(1.5, 2.5)[:, 0], np.arange(150, 250))


def test_live_session_tools():
    raw = SyntheticSource(n_units=4, duration_s=3.0, seed=8).read_raw(0, 3.0)
    src = FakeDevice(raw, buffer_s=5.0)
    sid = "live-test"
    server._sessions[sid] = src
    try:
        wait_until(lambda: server.get_stream_status(sid)["newest_sample_s"] >= 3.0)
        status = server.get_stream_status(sid)
        assert status["samples_received"] == 60_000 and status["samples_dropped"] == 0
        assert server.get_signal_quality(sid, t0=0.5, duration_s=1.0)["amplitude_unit"] == "ADC counts"
        assert server.get_firing_rates(sid)["n_units"] == 4
        with pytest.raises(ToolError, match="carries no behaviour|no velocity-like"):
            server.fit_decoder(sid)
    finally:
        server.close_session(sid)
    with pytest.raises(ToolError, match="not a live session"):
        server.get_stream_status(server.open_session("synthetic", {"duration_s": 2})["session_id"])


def test_lsl_round_trip():
    pylsl = pytest.importorskip("pylsl")
    sim = SyntheticSource(n_units=3, duration_s=2.0, seed=9)
    raw = sim.read_raw(0, 2.0)
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo("ephys-test", "EPHYS", 3, FS, pylsl.cf_float32, "ephys-test-1"))
    assert any(s["name"] == "ephys-test" for s in server.list_lsl_streams(wait_s=2.0)["streams"])
    sid = server.open_session("lsl", {"name": "ephys-test", "buffer_s": 5.0, "uv_per_count": 1.0})["session_id"]
    for i in range(0, len(raw), 5_000):
        outlet.push_chunk(raw[i : i + 5_000].tolist())
    wait_until(lambda: server.get_stream_status(sid)["samples_received"] >= len(raw))
    info = server.get_session_info(sid)
    assert info["n_channels"] == 3 and info["raw_fs_hz"] == FS and info["amplitude_unit"] == "uV"
    assert info["uri"].startswith("lsl://") and "'ephys-test'" in info["notes"]
    got = server._sessions[sid].read_raw(0.5, 1.0)
    np.testing.assert_allclose(got, raw[10_000:20_000], rtol=1e-6)
    server.close_session(sid)
    with pytest.raises(ToolError, match="no LSL stream"):
        server.open_session("lsl", {"name": "does-not-exist"})
