import numpy as np
import pytest
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.psth import compute_psth, usable_events


@pytest.fixture(scope="module")
def sid():
    return server.open_session("synthetic", {"n_units": 16, "duration_s": 300.0, "seed": 2})["session_id"]


@pytest.fixture(autouse=True)
def plots_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYS_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def test_psth_recovers_a_known_response():
    rng = np.random.default_rng(0)
    events = np.arange(1.0, 101.0)
    burst = np.concatenate([e + rng.uniform(0.1, 0.2, 5) for e in events])  # 50 Hz for 100 ms after each event
    psth = compute_psth([np.sort(burst)], events, (-0.2, 0.4), 0.1)
    assert psth.rate[0].tolist() == pytest.approx([0, 0, 0, 50, 0, 0])
    assert psth.n_events == 100


def test_events_with_unrecorded_windows_are_dropped():
    valid = np.array([[0.0, 10.0], [12.0, 20.0]])
    ok = usable_events(np.array([0.1, 5.0, 9.8, 11.0, 15.0, np.nan]), (-0.3, 0.6), valid)
    assert ok.tolist() == [False, True, False, False, True, False]


def test_single_unit_psth_shows_its_direction_tuning(sid):
    src = server._sessions[sid]
    unit = int(np.argmax(src.gain))
    res = server.get_psth(sid, group_by="direction", units=[unit], t_before=0.2, t_after=0.4)
    assert res["event"] == "move_onset_time" and set(res["groups"]) == {"down", "left", "right", "up"}
    px, py = src.preferred_dirs[unit]
    axis = ("right", "left") if abs(px) > abs(py) else ("up", "down")
    preferred, opposite = axis if (px if abs(px) > abs(py) else py) > 0 else axis[::-1]
    assert res["groups"][preferred]["peak_hz"] > res["groups"][opposite]["peak_hz"]


def test_psth_errors_are_readable(sid):
    with pytest.raises(ToolError, match="available"):
        server.get_psth(sid, event="nope_time")
    with pytest.raises(ToolError, match="cannot group by"):
        server.get_psth(sid, group_by="move_onset_time")
    with pytest.raises(ToolError, match="units must be in"):
        server.get_psth(sid, units=[999])


def test_plots_return_summary_and_png(sid, plots_dir):
    server.fit_decoder(sid, kind="ridge")
    results = [
        server.plot_psth(sid, group_by="direction"),
        server.plot_psth(sid, units=[0]),
        server.plot_raster(sid, t0=0.0, duration_s=5.0),
        server.plot_decoding(sid, t0=280.0, duration_s=10.0),
    ]
    for summary, image in results:
        path = plots_dir / summary["saved_to"].split("/")[-1]
        assert isinstance(image, Image) and path.read_bytes()[:4] == b"\x89PNG" and path.stat().st_size > 5_000
    assert results[3][0]["held_out"] is True


def test_plot_decoding_needs_a_decoder():
    sid = server.open_session("synthetic", {"n_units": 4, "duration_s": 30.0})["session_id"]
    with pytest.raises(ToolError, match="fit_decoder first"):
        server.plot_decoding(sid, t0=0.0)
