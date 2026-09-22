"""MCP server exposing read-only analysis of intracortical recordings.

Tools return compact summaries, never raw arrays, so results fit in an LLM
context window.
"""

from __future__ import annotations

import functools
import logging
import uuid
from typing import Any, Literal

import httpx
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import plots
from .processing.decode import DECODERS, r2_score
from .processing.psth import aligned_counts, compute_psth, modulation, summarize, usable_events
from .processing.quality import signal_quality
from .processing.rates import bin_spikes, firing_stats, resample_to
from .processing.spikes import detect_spikes as _detect_spikes
from .processing.spikes import match_spikes
from .sources import SOURCES, NeuralSource
from .sources import dandi as _dandi
from .sources import lsl as _lsl
from .sources.base import MAX_GROUPS
from .sources.live import RingBufferSource

DISCLAIMER = (
    "Research and education software. Not a medical device, not for clinical use, "
    "and not affiliated with or endorsed by any implant manufacturer."
)

mcp = MCPServer(
    "ephys-mcp",
    instructions=(
        "Read-only analysis of intracortical (spike-level) brain-computer-interface recordings. "
        "Start with list_sources, then open_session, then inspect with get_session_info. "
        "Tools return summaries rather than raw samples. " + DISCLAIMER
    ),
)

logging.getLogger("httpx").setLevel(logging.WARNING)

WARMUP_S = 1.0  # lead-in so stateful decoders converge before the requested window


def tool(fn=None, **options):
    """Register an MCP tool whose anticipated failures reach the model as readable messages."""
    if fn is None:
        return lambda f: tool(f, **options)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPError as exc:
            raise ToolError(f"DANDI Archive request failed: {exc}") from exc
        except (ValueError, KeyError, NotImplementedError) as exc:
            raise ToolError(str(exc.args[0]) if exc.args else type(exc).__name__) from exc

    mcp.tool(**options)(wrapper)
    return wrapper


_sessions: dict[str, NeuralSource] = {}
_decoders: dict[str, dict[str, Any]] = {}


def _session(session_id: str) -> NeuralSource:
    try:
        return _sessions[session_id]
    except KeyError:
        raise ValueError(f"unknown session_id {session_id!r}; call open_session first") from None


def _range(src: NeuralSource, t0: float, t1: float | None) -> tuple[float, float]:
    info = src.info()
    start, end = info.t_start_s, info.t_start_s + info.duration_s
    t1 = end if t1 is None else min(t1, end)
    t0 = max(t0, start)
    if not t0 < t1:
        raise ValueError(f"need {start} <= t0 < t1 <= {end}")
    return t0, t1


def _recorded_mask(src: NeuralSource, t: np.ndarray, bin_s: float) -> np.ndarray:
    """True for bins that lie wholly inside a span where data was recorded."""
    iv = src.valid_intervals()
    lo, hi = t - bin_s / 2, t + bin_s / 2
    k = np.clip(np.searchsorted(iv[:, 0], lo, side="right") - 1, 0, len(iv) - 1)
    return (lo >= iv[k, 0] - 1e-9) & (hi <= iv[k, 1] + 1e-9)


def _pick_target(src: NeuralSource, target: str | None) -> str:
    signals = src.info().behavior_signals
    if target:
        if target not in signals:
            raise ValueError(f"unknown target {target!r}; available: {sorted(signals)}")
        return target
    for name in signals:
        if "vel" in name.lower():
            return name
    raise ValueError(f"no velocity-like signal found; pass target explicitly from: {sorted(signals)}")


@tool
def list_sources() -> dict:
    """List the data source types this server can open, with their parameters and status."""
    return {
        "sources": [
            {"name": k, "status": v["status"], "description": v["description"], "default_params": v["params"]}
            for k, v in SOURCES.items()
        ],
        "disclaimer": DISCLAIMER,
    }


@tool
def open_session(source: str = "synthetic", params: dict[str, Any] | None = None) -> dict:
    """Open a recording session and return its session_id and metadata.

    `params` overrides the source's default_params (see list_sources).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; options: {sorted(SOURCES)}")
    src = SOURCES[source]["cls"](**{**SOURCES[source]["params"], **(params or {})})
    sid = uuid.uuid4().hex[:8]
    _sessions[sid] = src
    return {"session_id": sid, **src.info().to_dict()}


@tool
def close_session(session_id: str) -> dict:
    """Close a session and drop any decoder fitted on it."""
    _session(session_id).close()
    del _sessions[session_id]
    _decoders.pop(session_id, None)
    return {"closed": session_id}


@tool
def get_session_info(session_id: str) -> dict:
    """Metadata for a session: duration, channels, sampling rates, behaviour signals, licence, citation."""
    return _session(session_id).info().to_dict()


@tool
def get_signal_quality(session_id: str, t0: float = 0.0, duration_s: float = 2.0) -> dict:
    """Noise level, SNR and dead/noisy channels from a broadband snippet (max 10 s)."""
    src = _session(session_id)
    info = src.info()
    if info.raw_fs_hz is None:
        raise ValueError("this session has no broadband signal")
    raw = src.read_raw(t0, t0 + min(duration_s, 10.0))
    return {**signal_quality(raw, info.raw_fs_hz), "amplitude_unit": info.amplitude_unit}


@tool
def detect_spikes(session_id: str, t0: float = 0.0, duration_s: float = 2.0, threshold_sigma: float = -4.5) -> dict:
    """Threshold-crossing spike detection on a broadband snippet (max 10 s).

    Where the session has ground-truth spikes, precision and recall are reported.
    """
    src = _session(session_id)
    info = src.info()
    if info.raw_fs_hz is None:
        raise ValueError("this session has no broadband signal")
    t1 = t0 + min(duration_s, 10.0)
    spikes, sigma = _detect_spikes(src.read_raw(t0, t1), info.raw_fs_hz, threshold_sigma)
    counts = np.array([s.size for s in spikes])
    out = {
        "window_s": [t0, t1],
        "threshold_sigma": threshold_sigma,
        "total_spikes": int(counts.sum()),
        "median_rate_hz": round(float(np.median(counts) / (t1 - t0)), 2),
        "median_noise": round(float(np.median(sigma)), 2),
        "amplitude_unit": info.amplitude_unit,
    }
    if info.has_sorted_spikes:
        truth = src.spike_times(t0, t1)
        scores = [match_spikes(d + t0, tr) for d, tr in zip(spikes, truth)]
        out["vs_ground_truth"] = {
            "median_precision": round(float(np.median([s["precision"] for s in scores])), 3),
            "median_recall": round(float(np.median([s["recall"] for s in scores])), 3),
        }
    return out


@tool
def get_firing_rates(session_id: str, t0: float = 0.0, t1: float | None = None) -> dict:
    """Population firing-rate summary over a time range (defaults to the whole session)."""
    src = _session(session_id)
    t0, t1 = _range(src, t0, t1)
    iv = np.clip(src.valid_intervals(), t0, t1)
    recorded = float((iv[:, 1] - iv[:, 0]).sum())
    if recorded <= 0:
        raise ValueError("no recorded data in that window")
    stats = firing_stats(src.spike_times(t0, t1), 0.0, recorded)
    return {"window_s": [t0, t1], "recorded_s": round(recorded, 2), **stats}


def _xy(src: NeuralSource, target: str, t0: float, t1: float, bin_s: float):
    t, counts = bin_spikes(src.spike_times(t0, t1), t0, t1, bin_s)
    bt, bv = src.behavior(target, t0, t1)
    if bt.size < 2:
        raise ValueError("no behaviour samples in that window")
    y = resample_to(bt, bv, t)
    keep = _recorded_mask(src, t, bin_s) & np.isfinite(y).all(axis=1)
    if keep.sum() < 10:
        raise ValueError("too little recorded data in that window")
    return t[keep], counts[keep], y[keep]


@tool
def fit_decoder(
    session_id: str,
    kind: Literal["ridge", "kalman"] = "kalman",
    target: str | None = None,
    train_fraction: float = 0.8,
    bin_s: float = 0.05,
) -> dict:
    """Fit a decoder from spike counts to a behaviour signal and score it on held-out data.

    The first `train_fraction` of the session trains; the remainder tests. With no
    `target`, the first velocity-like behaviour signal is used. Unrecorded gaps are skipped.
    """
    src = _session(session_id)
    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must be in 0.1..0.95")
    if not 0.005 <= bin_s <= 1.0:
        raise ValueError("bin_s must be in 0.005..1.0")
    target = _pick_target(src, target)
    info = src.info()
    start, dur = info.t_start_s, info.t_start_s + info.duration_s
    split = start + info.duration_s * train_fraction
    _, Xtr, ytr = _xy(src, target, start, split, bin_s)
    _, Xte, yte = _xy(src, target, split, dur, bin_s)
    model = DECODERS[kind]().fit(Xtr, ytr)
    r2 = r2_score(yte, model.predict(Xte))
    _decoders[session_id] = {"model": model, "target": target, "bin_s": bin_s, "kind": kind, "test_start_s": split}
    return {
        "kind": kind,
        "params": model.params,
        "target": target,
        "bin_s": bin_s,
        "target_unit": info.behavior_units.get(target, ""),
        "n_train_bins": len(Xtr),
        "n_test_bins": len(Xte),
        "train_window_s": [start, round(split, 2)],
        "test_window_s": [round(split, 2), dur],
        "test_r2_per_dim": [round(float(v), 3) for v in r2],
        "test_r2_mean": round(float(r2.mean()), 3),
    }


@tool
def decode_window(session_id: str, t0: float, duration_s: float = 2.0, max_points: int = 20) -> dict:
    """Decode a window with the fitted decoder; returns a downsampled decoded-vs-true preview."""
    src = _session(session_id)
    if session_id not in _decoders:
        raise ValueError("no decoder fitted; call fit_decoder first")
    d = _decoders[session_id]
    t0, t1 = _range(src, t0, t0 + duration_s)
    lead = min(WARMUP_S, t0 - src.info().t_start_s)
    t, X, y = _xy(src, d["target"], t0 - lead, t1, d["bin_s"])
    keep = t >= t0
    t, y, yhat = t[keep], y[keep], d["model"].predict(X)[keep]
    step = max(1, len(t) // max(1, min(max_points, 100)))
    return {
        "kind": d["kind"],
        "target": d["target"],
        "window_r2_mean": round(float(r2_score(y, yhat).mean()), 3),
        "preview": [
            {"t": round(float(t[i]), 3), "decoded": np.round(yhat[i], 2).tolist(), "true": np.round(y[i], 2).tolist()}
            for i in range(0, len(t), step)
        ],
    }


PREFERRED_EVENTS = ("move_onset_time", "go_cue_time", "target_on_time")


def _aligned(
    src: NeuralSource,
    event: str | None,
    group_by: str | None,
    t_before: float,
    t_after: float,
    units: list[int] | None = None,
):
    """Usable event times per group label, the spikes covering them, and how many events were dropped."""
    if not (0 <= t_before <= 5 and 0 < t_after <= 5):
        raise ValueError("t_before must be in 0..5 s and t_after in (0, 5] s")
    trials = src.trials()
    info = src.info()
    if not info.event_columns:
        raise ValueError("this session has no trial events to align to")
    event = event or next((e for e in PREFERRED_EVENTS if e in trials), info.event_columns[0])
    if event not in info.event_columns:
        raise ValueError(f"unknown event {event!r}; available: {info.event_columns}")
    if group_by and group_by not in info.group_columns:
        raise ValueError(f"cannot group by {group_by!r}; columns with 2..{MAX_GROUPS} values: {info.group_columns}")
    times = np.asarray(trials[event], dtype=float)
    ok = usable_events(times, (-t_before, t_after), src.valid_intervals())
    labels = trials[group_by][ok] if group_by else np.full(int(ok.sum()), "all trials")
    times = times[ok]
    if times.size < 2:
        raise ValueError("fewer than 2 events have a fully recorded window; shorten t_before/t_after")
    groups = {str(g): times[labels == g] for g in np.unique(labels)}
    spikes = src.spike_times(times.min() - t_before, times.max() + t_after + 1e-9)
    if units:
        if min(units) < 0 or max(units) >= len(spikes):
            raise ValueError(f"units must be in 0..{len(spikes) - 1}")
        spikes = [spikes[u] for u in units]
    return event, groups, spikes, int((~ok).sum())


@tool
def get_psth(
    session_id: str,
    event: str | None = None,
    group_by: str | None = None,
    t_before: float = 0.3,
    t_after: float = 0.6,
    bin_s: float = 0.02,
    units: list[int] | None = None,
) -> dict:
    """Peri-stimulus time histogram: firing aligned to a trial event and averaged over trials.

    `event` is one of the session's event_columns (default: movement onset if present).
    `group_by` splits trials by one of its group_columns, e.g. reach direction.
    `units` restricts the analysis to some units; tuning that cancels in the population
    average (such as direction preference) only shows up per unit. Unit numbers in the
    result index into `units` when it is given.
    Returns the population time course, its peak, and the units that change most.
    """
    src = _session(session_id)
    if not 0.005 <= bin_s <= 0.5:
        raise ValueError("bin_s must be in 0.005..0.5")
    event, groups, spikes, dropped = _aligned(src, event, group_by, t_before, t_after, units)
    window = (-t_before, t_after)
    overall = compute_psth(spikes, np.concatenate(list(groups.values())), window, bin_s)
    out = {"event": event, "window_s": list(window), "bin_s": bin_s, "events_dropped_unrecorded": dropped}
    out.update(summarize(overall))
    if group_by:
        out["group_by"] = group_by
        out["groups"] = {}
        for label, times in groups.items():
            s = summarize(compute_psth(spikes, times, window, bin_s), max_points=0, top=0)
            out["groups"][label] = {k: s[k] for k in ("n_events", "baseline_hz", "peak_hz", "peak_time_s")}
    return out


def _figure(path, **summary) -> list:
    return [{"saved_to": str(path), **summary}, Image(path=path)]


@tool(structured_output=False)
def plot_psth(
    session_id: str,
    event: str | None = None,
    group_by: str | None = None,
    t_before: float = 0.3,
    t_after: float = 0.6,
    bin_s: float = 0.02,
    units: list[int] | None = None,
) -> list:
    """Figure: population PSTH per group (mean ± SEM over trials) above a unit-by-time heatmap of
    change from baseline. Same arguments as get_psth. Returns the PNG path and the image."""
    src = _session(session_id)
    if not 0.005 <= bin_s <= 0.5:
        raise ValueError("bin_s must be in 0.005..0.5")
    event, groups, spikes, _ = _aligned(src, event, group_by, t_before, t_after, units)
    window = (-t_before, t_after)
    curves = {}
    for label, times in groups.items():
        t, counts = aligned_counts(spikes, times, window, bin_s)
        pop = counts.mean(axis=1) / bin_s  # population rate per trial
        sem = pop.std(axis=0, ddof=1) / np.sqrt(len(times)) if len(times) > 1 else np.zeros(pop.shape[1])
        curves[label] = (pop.mean(axis=0), sem, len(times))
    overall = compute_psth(spikes, np.concatenate(list(groups.values())), window, bin_s)
    pre = overall.t < 0
    base = overall.rate[:, pre].mean(axis=1, keepdims=True) if pre.any() else overall.rate.mean(axis=1, keepdims=True)
    sd = (overall.rate[:, pre].std(axis=1, keepdims=True) if pre.any() else 0.0) + 1.0
    n = sum(len(v) for v in groups.values())
    who = f"unit {units[0]}" if units and len(units) == 1 else f"{len(spikes)} units"
    title = f"Firing around {event} · {n} trials, {who}"
    rate_label = "Firing rate (Hz)" if len(spikes) == 1 else "Mean rate per unit (Hz)"
    path = plots.plot_psth(f"{session_id}-psth", title, event, t, curves, (overall.rate - base) / sd, rate_label)
    top = np.argsort(-np.abs(modulation(overall)))[:5]
    return _figure(path, event=event, n_events=n, groups=list(curves), most_modulated_units=[int(u) for u in top])


@tool(structured_output=False)
def plot_raster(session_id: str, t0: float = 0.0, duration_s: float = 10.0, max_units: int = 150) -> list:
    """Figure: spike raster for a time window (max 60 s), with unrecorded spans shaded."""
    src = _session(session_id)
    t0, t1 = _range(src, t0, t0 + min(max(duration_s, 0.1), 60.0))
    spikes = src.spike_times(t0, t1)[: max(1, min(max_units, 400))]
    title = f"Spike raster · {len(spikes)} units, {t0:g}–{t1:g} s"
    path = plots.plot_raster(f"{session_id}-raster", title, spikes, t0, t1, src.valid_intervals())
    return _figure(path, window_s=[t0, t1], n_units=len(spikes), total_spikes=int(sum(s.size for s in spikes)))


@tool(structured_output=False)
def plot_decoding(session_id: str, t0: float, duration_s: float = 10.0) -> list:
    """Figure: decoded against actual behaviour for a window (max 60 s), one panel per dimension.
    Requires fit_decoder first. Pick a window in the test split for an honest picture."""
    src = _session(session_id)
    if session_id not in _decoders:
        raise ValueError("no decoder fitted; call fit_decoder first")
    d = _decoders[session_id]
    t0, t1 = _range(src, t0, t0 + min(max(duration_s, 0.5), 60.0))
    lead = min(WARMUP_S, t0 - src.info().t_start_s)
    t, X, y = _xy(src, d["target"], t0 - lead, t1, d["bin_s"])
    keep = t >= t0
    t, y, yhat = t[keep], y[keep], d["model"].predict(X)[keep]
    r2 = r2_score(y, yhat)
    in_test = t0 >= d["test_start_s"]
    title = (
        f"{d['kind'].capitalize()} decoder · {d['target']} · {'held-out data' if in_test else 'includes training data'}"
    )
    unit = src.info().behavior_units.get(d["target"], "")
    path = plots.plot_decode(f"{session_id}-decode", title, t, y, yhat, unit, r2)
    return _figure(path, window_s=[t0, t1], r2_per_dim=[round(float(v), 3) for v in r2], held_out=bool(in_test))


@tool
def list_lsl_streams(wait_s: float = 3.0) -> dict:
    """Lab Streaming Layer streams visible on the local network (needs the lsl extra)."""
    streams = _lsl.list_streams(max(0.5, min(wait_s, 10.0)))
    return {
        "streams": streams,
        "hint": "open_session(source='lsl', params={'name': ...}) to subscribe" if streams else "none found",
    }


@tool
def get_stream_status(session_id: str) -> dict:
    """For a live session: how much signal is buffered, whether data is still arriving, and drops."""
    src = _session(session_id)
    if not isinstance(src, RingBufferSource):
        raise ValueError("not a live session; this tool applies to lsl sessions")  # noqa: TRY004 - a bad argument, not a type bug
    return src.status()


@tool
def search_datasets(query: str = "", limit: int = 10) -> dict:
    """Search the DANDI Archive for datasets. An empty query returns a short curated list of
    intracortical datasets that work well with this server."""
    if not query.strip():
        return {"curated": _dandi.CATALOG}
    return {"results": _dandi.search(query, limit), "curated": _dandi.CATALOG}


@tool
def list_dataset_files(dandiset_id: str, version: str = "", limit: int = 50) -> dict:
    """Licence, citation and NWB files of a DANDI dataset. Pass a file's path to open_session(source='dandi')."""
    meta = _dandi.describe(dandiset_id, version or None)
    return {**meta, "files": _dandi.list_files(dandiset_id, meta["version"], limit)}


@mcp.resource("ephys://sessions")
def sessions_resource() -> dict:
    """Currently open sessions."""
    return {sid: s.info().to_dict() for sid, s in _sessions.items()}


@mcp.prompt()
def analyze_session(session_id: str) -> str:
    """Walk through a standard first-pass analysis of a session."""
    return (
        f"Analyse session {session_id}: get_session_info, then get_signal_quality, then get_firing_rates. "
        "If a cursor or hand velocity signal exists, fit_decoder with kind='kalman' and with kind='ridge', "
        "compare held-out R², and plot_decoding on a test segment. If the session has trial events, "
        "get_psth and plot_psth grouped by a sensible column. Summarise findings and caveats."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
