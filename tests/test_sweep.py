"""Regressions from the final review sweep."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries

from ephys_mcp import server
from ephys_mcp.sources import NwbSource, SyntheticSource


def test_broadband_only_nwb_with_electrodes_opens():
    nwb = NWBFile(session_description="bb", identifier="bb", session_start_time=datetime.now(UTC))
    dev = nwb.create_device("d")
    grp = nwb.create_electrode_group("g", description="g", location="M1", device=dev)
    for i in range(4):
        nwb.add_electrode(x=i * 4e-4, y=0.0, z=0.0, location="M1", group=grp, filtering="none")
    region = nwb.create_electrode_table_region(list(range(4)), "all")
    data = (np.random.default_rng(0).standard_normal((20_000, 4)) * 10).astype(np.int16)
    nwb.add_acquisition(ElectricalSeries(name="raw", data=data, electrodes=region, rate=20_000.0, conversion=1e-6))
    path = Path(tempfile.mkdtemp()) / "bb.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    src = NwbSource(str(path))
    info = src.info()
    assert info.n_channels == 4 and info.raw_fs_hz == 20_000.0 and not info.has_sorted_spikes
    assert info.has_probe_geometry and src.channel_positions()[1].tolist() == [400.0, 0.0]  # metres -> um
    assert info.channel_areas == {"M1": 4}
    assert src.read_raw(0, 0.5).shape == (10_000, 4)
    src.close()


def test_open_session_rejects_unknown_params_readably():
    with pytest.raises(ToolError, match=r"unknown params \['bogus'\].*valid"):
        server.open_session("synthetic", {"bogus": 1})


def test_sort_spikes_validates_channels():
    sid = server.open_session("synthetic", {"n_units": 4, "duration_s": 10.0})["session_id"]
    with pytest.raises(ToolError, match="channels must be"):
        server.sort_spikes(sid, duration_s=5.0, channels=[99])
    server.close_session(sid)


def test_synthetic_noise_is_consistent_across_overlapping_reads():
    src = SyntheticSource(n_units=3, duration_s=5.0, seed=2)
    a, b = src.read_raw(0.0, 2.0), src.read_raw(1.0, 3.0)
    np.testing.assert_array_equal(a[20_000:], b[:20_000])


def test_decoder_mask_carries_into_decode_window():
    class Masked(SyntheticSource):
        def info(self):
            i = super().info()
            i.behavior_signals["moving"] = 1
            return i

        def behavior(self, name, t0, t1):
            if name != "moving":
                return super().behavior(name, t0, t1)
            t, v = super().behavior("cursor_velocity", t0, t1)
            return t, (np.linalg.norm(v, axis=1) > 8.0).astype(float)[:, None]

    sid = "sweep-masked"
    server._sessions[sid] = Masked(n_units=16, duration_s=120.0, seed=3)
    try:
        server.fit_decoder(sid, kind="ridge", mask="moving")
        masked = server.decode_window(sid, t0=100.0, duration_s=10.0, max_points=100)
        assert masked["mask"] == "moving"
        server.fit_decoder(sid, kind="ridge")
        full = server.decode_window(sid, t0=100.0, duration_s=10.0, max_points=100)
        assert full["mask"] is None
        # 10 s at 50 ms bins is 200 bins; the mask must have removed some of them
        assert len(masked["preview"]) < 200 <= len(full["preview"]) * 2
    finally:
        server._sessions.pop(sid)


@pytest.mark.skipif(not pytest.importorskip("spikeinterface", reason="sort extra"), reason="sort extra")
def test_sorted_session_is_the_sorted_window_and_keeps_no_stale_state():
    sid = server.open_session("synthetic", {"n_units": 6, "duration_s": 60.0, "seed": 5, "pitch_um": 20.0})[
        "session_id"
    ]
    server.fit_latent_factors(sid, n_factors=2, method="pca")
    server.fit_decoder(sid, kind="ridge")
    server.sort_spikes(sid, t0=10.0, duration_s=20.0)
    info = server.get_session_info(sid)
    assert info["t_start_s"] == 10.0 and info["duration_s"] == 20.0
    assert "vs_ground_truth" not in server.detect_spikes(sid, t0=12.0)  # sorter output is not ground truth
    with pytest.raises(ToolError, match="fit_latent_factors first"):
        server.plot_latent_factors(sid)
    with pytest.raises(ToolError, match="fit_decoder first"):
        server.decode_window(sid, t0=12.0)
    probe = server.get_probe(sid)
    assert probe["n_channels"] == info["n_channels"] and probe["geometry_source"] == "file"
    server.close_session(sid)
