from datetime import UTC, datetime

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries

from ephys_mcp.sources import NwbSource


@pytest.fixture
def falcon_like(tmp_path):
    """Behaviour in acquisition, a boolean mask, and the sampling *period* stored in `rate`."""
    nwb = NWBFile(session_description="acq fixture", identifier="acq", session_start_time=datetime.now(UTC))
    n = 3000  # 60 s at 50 Hz
    for u in range(4):
        nwb.add_unit(spike_times=np.sort(np.random.default_rng(u).uniform(0, 60, 600)))
    nwb.add_acquisition(TimeSeries(name="kin_vel", data=np.zeros((n, 3)), rate=0.02, unit="arbitrary"))
    nwb.add_acquisition(TimeSeries(name="eval_mask", data=np.arange(n) % 2 == 0, rate=0.02, unit="bool"))
    path = tmp_path / "acq.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return str(path)


def test_acquisition_behaviour_and_period_as_rate(falcon_like):
    src = NwbSource(falcon_like)
    info = src.info()
    assert info.behavior_signals == {"kin_vel": 3, "eval_mask": 1}
    assert info.behavior_fs_hz == 50.0  # 1/0.02, not 0.02
    assert "sampling period instead of a rate" in info.notes
    t, v = src.behavior("eval_mask", 0, 60)
    assert t[-1] == pytest.approx(59.98) and v.shape == (n_expected := 3000, 1) and v[:, 0].sum() == n_expected / 2
    src.close()
