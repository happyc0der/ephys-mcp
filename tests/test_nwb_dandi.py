from datetime import UTC, datetime

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pynwb import NWBHDF5IO, NWBFile, TimeSeries

from ephys_mcp import server
from ephys_mcp.sources import NwbSource, SyntheticSource, dandi

GAP = (20.0, 25.0)  # nothing recorded here


@pytest.fixture(scope="module")
def nwb_path(tmp_path_factory):
    """A small NWB file built from the simulator, with an unrecorded gap like trial-based datasets have."""
    sim = SyntheticSource(n_units=24, duration_s=60.0, seed=3)
    nwb = NWBFile(session_description="generated fixture", identifier="fixture", session_start_time=datetime.now(UTC))
    obs = [[0.0, GAP[0]], [GAP[1], 60.0]]
    for spikes in sim.spike_times(0, 60):
        nwb.add_unit(spike_times=spikes[(spikes < GAP[0]) | (spikes >= GAP[1])], obs_intervals=obs)
    t, vel = sim.behavior("cursor_velocity", 0, 60)
    keep = (t < GAP[0]) | (t >= GAP[1])
    mod = nwb.create_processing_module("behavior", "behaviour")
    mod.add(TimeSeries(name="hand_vel", data=vel[keep], timestamps=t[keep], unit="cm/s"))
    path = tmp_path_factory.mktemp("nwb") / "fixture.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return str(path)


def test_nwb_info_and_gaps(nwb_path):
    src = NwbSource(nwb_path)
    info = src.info()
    assert info.n_channels == 24 and info.raw_fs_hz is None and info.has_sorted_spikes
    assert info.behavior_signals == {"hand_vel": 2} and info.behavior_units == {"hand_vel": "cm/s"}
    assert info.recorded_fraction == pytest.approx(55 / 60, abs=0.01)
    assert src.valid_intervals().tolist() == [[0.0, 20.0], [25.0, 60.0]]
    with pytest.raises(NotImplementedError):
        src.read_raw(0, 1)
    src.close()


def test_nwb_rejects_bad_paths(tmp_path):
    for bad in ("https://example.com/x.nwb", str(tmp_path / "missing.nwb"), __file__):
        with pytest.raises(ValueError):
            NwbSource(bad)


def test_decoding_skips_unrecorded_gap(nwb_path):
    sid = server.open_session("nwb", {"path": nwb_path})["session_id"]
    t, _, _ = server._xy(server._sessions[sid], "hand_vel", 0.0, 60.0, 0.05)
    assert not ((t > GAP[0]) & (t < GAP[1])).any()
    assert server.get_firing_rates(sid)["recorded_s"] == pytest.approx(55.0)
    res = server.fit_decoder(sid, kind="ridge")  # target auto-selected
    assert res["target"] == "hand_vel" and res["test_r2_mean"] > 0.5
    with pytest.raises(ToolError, match="no broadband"):
        server.get_signal_quality(sid)
    with pytest.raises(ToolError, match="available"):
        server.fit_decoder(sid, target="nope")
    server.close_session(sid)


def _fake_dandi(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/dandisets/"):
        version = {"version": "0.1", "name": "Demo", "asset_count": 1, "size": 2_000_000_000}
        return httpx.Response(
            200, json={"results": [{"identifier": "000001", "most_recent_published_version": version}]}
        )
    if path.endswith("/dandisets/000001/"):
        return httpx.Response(200, json={"most_recent_published_version": {"version": "0.1"}})
    if path.endswith("/versions/0.1/"):
        return httpx.Response(200, json={"name": "Demo", "license": ["spdx:CC-BY-4.0"], "citation": "Doe (2026)"})
    if path.endswith("/assets/"):
        return httpx.Response(200, json={"results": [{"path": "sub-1/a.nwb", "size": 1_500_000, "asset_id": "abc"}]})
    return httpx.Response(404)


def test_dandi_tools_with_mocked_api(monkeypatch):
    fake = httpx.Client(base_url=dandi.API, transport=httpx.MockTransport(_fake_dandi))
    monkeypatch.setattr(dandi, "_client", fake)
    assert server.search_datasets("demo")["results"][0] == {
        "dandiset_id": "000001",
        "name": "Demo",
        "version": "0.1",
        "n_files": 1,
        "size_gb": 2.0,
    }
    files = server.list_dataset_files("000001")
    assert files["license"] == "CC-BY-4.0" and files["citation"] == "Doe (2026)"
    assert files["files"] == [{"path": "sub-1/a.nwb", "size_mb": 1.5, "asset_id": "abc"}]
    assert "curated" in server.search_datasets("")
    with pytest.raises(ToolError, match="six digits"):
        server.list_dataset_files("../etc")
    with pytest.raises(ToolError, match="not found"):
        server.list_dataset_files("999999")


@pytest.mark.network
def test_real_dandi_session():
    sid = server.open_session("dandi", {"dandiset_id": "000140"})["session_id"]
    info = server.get_session_info(sid)
    assert info["license"] == "CC-BY-4.0" and "Churchland" in info["citation"] and info["n_channels"] == 142
    assert server.fit_decoder(sid, kind="ridge")["test_r2_mean"] > 0.4
    assert server.fit_decoder(sid, kind="kalman")["test_r2_mean"] > 0.25
    server.close_session(sid)
