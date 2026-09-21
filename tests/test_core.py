import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.spikes import detect_spikes, match_spikes
from ephys_mcp.sources import N1StubSource, SyntheticSource


@pytest.fixture(scope="module")
def src():
    return SyntheticSource(n_units=16, duration_s=60.0, seed=1)


def test_synthetic_shapes(src):
    info = src.info()
    assert info.n_channels == 16 and info.raw_fs_hz == 20_000
    raw = src.read_raw(1.0, 1.5, channels=[0, 3])
    assert raw.shape == (10_000, 2)
    t, v = src.behavior("cursor_velocity", 0, 10)
    assert v.shape == (t.size, 2)


def test_raw_read_is_bounded(src):
    with pytest.raises(ValueError):
        src.read_raw(0, 11)


def test_spike_detection_matches_truth(src):
    t0, t1 = 5.0, 9.0
    detected, _ = detect_spikes(src.read_raw(t0, t1), 20_000)
    truth = src.spike_times(t0, t1)
    scores = [match_spikes(d + t0, tr) for d, tr in zip(detected, truth)]
    assert np.median([s["precision"] for s in scores]) > 0.9
    assert np.median([s["recall"] for s in scores]) > 0.85


@pytest.mark.parametrize("kind", ["ridge", "kalman"])
def test_decoder_recovers_velocity(kind):
    sid = server.open_session("synthetic", {"n_units": 32, "duration_s": 120.0})["session_id"]
    res = server.fit_decoder(sid, kind=kind)
    assert res["test_r2_mean"] > 0.6
    preview = server.decode_window(sid, t0=100.0, duration_s=2.0)
    assert 0 < len(preview["preview"]) <= 40
    server.close_session(sid)


def test_tool_errors_are_readable():
    with pytest.raises(ToolError, match="unknown session_id"):
        server.get_session_info("nope")
    with pytest.raises(ToolError, match="No public device API"):
        server.open_session("n1_stub")


def test_device_stub_is_not_implemented():
    with pytest.raises(NotImplementedError):
        N1StubSource()


@pytest.mark.asyncio
async def test_tools_are_registered():
    names = {t.name for t in await server.mcp.list_tools()}
    assert {"list_sources", "open_session", "fit_decoder", "decode_window", "get_signal_quality"} <= names
    assert not any("stim" in n or "write" in n for n in names)
