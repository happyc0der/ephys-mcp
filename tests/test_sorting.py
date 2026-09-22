import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.spikes import match_spikes

pytest.importorskip("spikeinterface")


def test_sorting_recovers_units_and_replaces_session_spikes(tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYS_MCP_OUTPUT_DIR", str(tmp_path))
    sid = server.open_session("synthetic", {"n_units": 6, "duration_s": 40.0, "seed": 12})["session_id"]
    truth_src = server._sessions[sid]
    res = server.sort_spikes(sid, t0=0.0, duration_s=30.0)
    assert res["sorter"] == "spykingcircus2" and res["window_s"] == [0.0, 30.0]
    assert 5 <= res["n_units"] <= 7 and res["n_units_clean_isi"] >= 5
    assert {u["channel"] for u in res["units"]} >= set(range(5))
    assert all(u["peak_amplitude"] < -40 for u in res["units"])  # spikes are -80 uV in the simulator

    sorted_src = server._sessions[sid]
    detected, truth = sorted_src.spike_times(0, 30), truth_src.spike_times(0, 30)
    best_recall = [max(match_spikes(d, t)["recall"] for d in detected) for t in truth]
    assert np.median(best_recall) > 0.95

    info = server.get_session_info(sid)
    assert info["has_sorted_spikes"] and info["n_channels"] == res["n_units"] and "sorted by" in info["notes"]
    rates = server.get_firing_rates(sid)
    assert rates["recorded_s"] == pytest.approx(30.0) and rates["n_units"] == res["n_units"]
    assert server.plot_raster(sid, t0=0.0, duration_s=5.0)[0]["n_units"] == res["n_units"]
    assert server.get_psth(sid)["n_events"] > 0
    server.close_session(sid)


def test_sorting_rejects_sessions_without_broadband_and_huge_windows():
    sid = server.open_session("synthetic", {"n_units": 64, "duration_s": 200.0})["session_id"]
    with pytest.raises(ToolError, match="channel-seconds"):
        server.sort_spikes(sid, duration_s=200.0)
    server.close_session(sid)
