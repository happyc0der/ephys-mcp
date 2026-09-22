import numpy as np
import pytest
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError

from ephys_mcp import server
from ephys_mcp.processing.latent import fit_gpfa, fit_pca


@pytest.fixture(scope="module")
def sid():
    return server.open_session("synthetic", {"n_units": 24, "duration_s": 240.0, "seed": 4})["session_id"]


def test_gpfa_recovers_a_planted_two_dimensional_latent():
    rng = np.random.default_rng(0)
    n_trials, T, n_units, bin_s = 60, 40, 20, 0.02
    t = np.arange(T) * bin_s
    x = np.stack(
        [
            np.stack(
                [
                    np.sin(2 * np.pi * t / 0.8 + rng.uniform(0, 2 * np.pi)),
                    np.cos(2 * np.pi * t / 0.5 + rng.uniform(0, 2 * np.pi)),
                ],
                axis=1,
            )
            for _ in range(n_trials)
        ]
    )
    C = rng.standard_normal((n_units, 2))
    rate = np.exp(1.0 + 0.6 * x @ C.T)  # (trials, T, units), ~3-10 Hz in 20 ms bins scale
    counts = rng.poisson(rate * 0.5)
    fit = fit_gpfa(counts, 4, bin_s)
    ve = fit.variance_explained
    assert ve[0] + ve[1] > 0.85 and ve[2] < 0.1  # elbow at two factors
    assert fit.log_likelihood[-1] > fit.log_likelihood[0]  # EM increased the likelihood
    assert np.all(np.diff(fit.log_likelihood) > -1e-6 * abs(fit.log_likelihood[0]))  # and never decreased
    X = np.c_[fit.trajectories[:, :, :2].reshape(-1, 2), np.ones(n_trials * T)]
    W = np.linalg.lstsq(X, x.reshape(-1, 2), rcond=None)[0]
    resid = x.reshape(-1, 2) - X @ W
    assert 1 - resid.var(axis=0).sum() / x.reshape(-1, 2).var(axis=0).sum() > 0.85
    assert all(0.05 < tau < 0.5 for tau in fit.timescales_s[:2])


def test_pca_shapes_and_ordering():
    counts = np.random.default_rng(1).poisson(2.0, size=(10, 30, 12))
    fit = fit_pca(counts, 5)
    assert fit.trajectories.shape == (10, 30, 5) and fit.loading.shape == (12, 5)
    assert np.all(np.diff(fit.variance_explained) <= 0) and fit.variance_explained.sum() < 1


def test_latent_tool_explains_cursor_velocity(sid, tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYS_MCP_OUTPUT_DIR", str(tmp_path))
    res = server.fit_latent_factors(sid, group_by="direction", n_factors=5)
    assert res["method"] == "gpfa" and res["event"] == "move_onset_time"
    ve = res["variance_explained_per_factor"]
    assert ve[0] + ve[1] > 0.8  # the simulator's latent is 2-D cursor velocity
    assert res["behaviour_explained"]["signal"] == "cursor_velocity"
    assert res["behaviour_explained"]["r2_top2"] > 0.85
    assert res["behaviour_explained"]["r2_top2"] > res["behaviour_explained"]["r2_top1"] + 0.2
    summary, image = server.plot_latent_factors(sid)
    assert isinstance(image, Image) and summary["groups"] == ["down", "left", "right", "up"]
    pca = server.fit_latent_factors(sid, method="pca", n_factors=5)
    assert pca["behaviour_explained"]["r2_top2"] < res["behaviour_explained"]["r2_top2"]  # GPFA denoises


def test_latent_tool_with_fewer_than_three_factors(sid):
    res = server.fit_latent_factors(sid, n_factors=2, method="pca")
    assert set(res["behaviour_explained"]) == {"signal", "r2_top1", "r2_top2"}


def test_latent_tool_errors(sid):
    with pytest.raises(ToolError, match="below the number of units"):
        server.fit_latent_factors(sid, n_factors=20, units=[0, 1, 2])
    fresh = server.open_session("synthetic", {"n_units": 4, "duration_s": 30.0})["session_id"]
    with pytest.raises(ToolError, match="fit_latent_factors first"):
        server.plot_latent_factors(fresh)
    server.close_session(fresh)
