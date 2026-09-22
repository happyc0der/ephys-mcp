import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.sources import SyntheticSource


class DriftedSource(SyntheticSource):
    """A later 'day' on the same array: day 0's units and tuning, a new trajectory, and
    optionally a rotation of each unit's preferred direction."""

    def __init__(self, day0: SyntheticSource, seed_day: int, drift: float):
        super().__init__(n_units=day0.n_units, duration_s=day0.duration_s, noise=day0.noise, seed=seed_day)
        rng = np.random.default_rng(seed_day)
        self.baseline_hz, self.gain = day0.baseline_hz, day0.gain
        theta = np.arctan2(day0.preferred_dirs[:, 1], day0.preferred_dirs[:, 0])
        theta = theta + drift * rng.standard_normal(self.n_units)
        self.preferred_dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._regenerate(rng)

    def _regenerate(self, rng):
        from ephys_mcp.sources.synthetic import BEHAVIOR_FS, REFRACTORY_S

        rate = self.baseline_hz + self.gain * (self._vel @ self.preferred_dirs.T)
        rate = np.clip(rate, 0.5, None)
        counts = rng.poisson(rate / BEHAVIOR_FS)
        self._spikes = []
        for u in range(self.n_units):
            idx = np.repeat(np.arange(len(rate)), counts[:, u])
            t = np.sort((idx + rng.uniform(0, 1, idx.size)) / BEHAVIOR_FS)
            keep, last = np.ones(t.size, bool), -np.inf
            for i, ti in enumerate(t):
                if ti - last < REFRACTORY_S:
                    keep[i] = False
                else:
                    last = ti
            self._spikes.append(t[keep])


@pytest.fixture(scope="module")
def days():
    base = {"n_units": 24, "duration_s": 120.0, "noise": 0.5}
    ids = {"day0": server.open_session("synthetic", {**base, "seed": 20})["session_id"]}
    day0 = server._sessions[ids["day0"]]
    for name, drift in (("same_tuning", 0.0), ("drifted", 1.2)):
        src = DriftedSource(day0, 99, drift)
        sid = f"cs-{name}"
        server._sessions[sid] = src
        ids[name] = sid
    yield ids
    for sid in ids.values():
        server._sessions.pop(sid, None)


def test_transfer_degrades_with_tuning_drift(days):
    res = server.evaluate_cross_session(days["day0"], [days["same_tuning"], days["drifted"]], kind="ridge", bin_s=0.05)
    assert res["mask"] is None and res["target"] == "cursor_velocity" and res["n_units"] == 24
    assert res["params"]["history_s"] == 0.5
    same, drifted = res["transfer"]
    assert res["train_tail_r2_mean"] > 0.6
    assert same["r2_mean"] > 0.6  # same units, same tuning, different day: transfers
    assert drifted["r2_mean"] < same["r2_mean"] - 0.2  # tuning changed: transfer degrades


class MaskedSource(SyntheticSource):
    """A synthetic session with an extra boolean signal: true while the cursor is moving fast."""

    def info(self):
        i = super().info()
        i.behavior_signals["moving"] = 1
        i.behavior_units["moving"] = "bool"
        return i

    def behavior(self, name, t0, t1):
        if name != "moving":
            return super().behavior(name, t0, t1)
        t, v = super().behavior("cursor_velocity", t0, t1)
        return t, (np.linalg.norm(v, axis=1) > 8.0).astype(float)[:, None]


def test_mask_restricts_scored_bins():
    sid = "cs-masked"
    server._sessions[sid] = MaskedSource(n_units=16, duration_s=120.0, seed=22)
    try:
        masked = server.fit_decoder(sid, kind="ridge", mask="moving")
        full = server.fit_decoder(sid, kind="ridge")
        assert masked["mask"] == "moving" and masked["n_test_bins"] < full["n_test_bins"]
        with pytest.raises(ToolError, match="unknown mask"):
            server.fit_decoder(sid, mask="nope")
    finally:
        server._sessions.pop(sid)


def test_cross_session_rejects_mismatched_units(days):
    other = server.open_session("synthetic", {"n_units": 8, "duration_s": 20.0})["session_id"]
    with pytest.raises(ToolError, match="has 8 units"):
        server.evaluate_cross_session(days["day0"], [other])
    with pytest.raises(ToolError, match="at least one"):
        server.evaluate_cross_session(days["day0"], [])
    server.close_session(other)
