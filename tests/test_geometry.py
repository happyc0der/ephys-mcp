import numpy as np
import pytest
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.geometry import make_layout, summarize
from ephys_mcp.sources import SyntheticSource


def test_layouts():
    utah = make_layout("utah", 96, 400.0)
    assert utah.shape == (96, 2) and summarize(utah)["nearest_neighbour_um"] == 400.0 and not summarize(utah)["dense"]
    lin = make_layout("linear", 8, 20.0)
    assert lin[:, 0].tolist() == [0] * 8 and lin[-1, 1] == 140.0 and summarize(lin)["dense"]
    grid = make_layout("grid", 6, 50.0, n_columns=3)
    assert grid.tolist() == [[0, 0], [50, 0], [100, 0], [0, 50], [50, 50], [100, 50]]
    tet = make_layout("tetrode_linear", 8, 200.0)
    assert summarize(tet)["nearest_neighbour_um"] == 25.0
    with pytest.raises(ValueError):
        make_layout("hex", 4, 10.0)


def test_dense_simulator_bleeds_units_onto_neighbours():
    src = SyntheticSource(n_units=6, duration_s=5.0, seed=1, pitch_um=20.0)
    assert src.info().has_probe_geometry and src.channel_positions().shape == (6, 2)
    raw = src.read_raw(0, 2.0)
    spikes = src.spike_times(0, 2.0)[2]  # unit 2 lives on contact 2
    idx = (spikes * 20_000).astype(int)
    at_spikes = np.array([raw[max(0, i - 10) : i + 20].min(axis=0) for i in idx[:50]]).mean(axis=0)
    assert at_spikes[2] < at_spikes[1] < at_spikes[0]  # deepest on its own contact, decaying outward
    assert at_spikes[2] < at_spikes[3] < at_spikes[4]


def test_probe_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYS_MCP_OUTPUT_DIR", str(tmp_path))
    sid = server.open_session("synthetic", {"n_units": 16, "duration_s": 20.0})["session_id"]
    assert server.get_probe(sid)["geometry_source"] is None and "hint" in server.get_probe(sid)
    with pytest.raises(ToolError, match="set_probe_geometry first"):
        server.plot_probe(sid)
    res = server.set_probe_geometry(sid, layout="grid", pitch_um=400.0, n_columns=4)
    assert res["n_contacts"] == 16 and res["span_um"] == [1200.0, 1200.0]
    probe = server.get_probe(sid)
    assert probe["geometry_source"] == "user" and probe["channels"][5] == {"channel": 5, "x_um": 400.0, "y_um": 400.0}
    summary, image = server.plot_probe(sid, duration_s=5.0)
    assert isinstance(image, Image) and summary["n_contacts"] == 16
    with pytest.raises(ToolError, match="16 finite"):
        server.set_probe_geometry(sid, layout="explicit", positions_um=[[0, 0]])
    server.close_session(sid)


@pytest.mark.skipif(not pytest.importorskip("spikeinterface", reason="sort extra"), reason="sort extra")
def test_geometry_prevents_duplicate_units_on_a_dense_probe():
    sid = server.open_session("synthetic", {"n_units": 12, "duration_s": 40.0, "seed": 5, "pitch_um": 20.0})[
        "session_id"
    ]
    with_geometry = server.sort_spikes(sid, duration_s=30.0)
    assert with_geometry["geometry"] == "used" and with_geometry["n_units"] == 12
    assert sorted(u["channel"] for u in with_geometry["units"]) == list(range(12))
    assert with_geometry["units"][0]["position_um"] is not None
    server.set_probe_geometry(sid, layout="linear", pitch_um=1000.0)  # pretend the contacts are isolated
    isolated = server.sort_spikes(sid, duration_s=30.0)
    assert isolated["n_units"] > 12  # the same units get counted again on neighbouring contacts
    server.close_session(sid)
