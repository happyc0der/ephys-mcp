"""Static figures, saved as PNG. Colour is assigned by job: categorical hues in a
fixed order for groups, a blue-gray-red diverging map for signed change, ink for
single series. The categorical set is validated for colour-blind separation."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#86b6ef", "#f0efec", "#ec835a", "#a82a2a"])
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
UNRECORDED = "#d5d3c8"
MAX_DIRECT_LABELS = 4

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
    }
)


def output_dir() -> Path:
    d = Path(os.environ.get("EPHYS_MCP_OUTPUT_DIR", Path.home() / ".cache" / "ephys-mcp" / "plots")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, stem: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", stem)
    path = output_dir() / f"{safe}-{int(time.time() * 1000) % 10**9}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _event_line(ax):
    ax.axvline(0, color=INK_2, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)


def _title_and_legend(ax, title: str, legend_cols: int):
    """Title on top, legend in its own band beneath it, so neither ever covers data."""
    if legend_cols:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncols=legend_cols,
            handlelength=1.4,
            columnspacing=1.2,
            borderpad=0,
        )
        rows = -(-len(legend.get_texts()) // legend_cols)
        ax.set_title(title, pad=10 + 15 * rows)
    else:
        ax.set_title(title)


def plot_psth(
    stem: str, title: str, event: str, t, groups: dict[str, tuple], unit_z: np.ndarray, rate_label: str
) -> Path:
    """groups: label -> (mean, sem, n) of the population rate. unit_z: (n_units, n_bins) baseline z-scores."""
    heatmap = len(unit_z) > 1  # a one-row heatmap says nothing the curve above does not
    if heatmap:
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.16}
        )
    else:
        fig, top = plt.subplots(figsize=(7.2, 3.6))
    for (label, (mean, sem, n)), colour in zip(groups.items(), CATEGORICAL):
        top.fill_between(t, mean - sem, mean + sem, color=colour, alpha=0.16, linewidth=0)
        top.plot(t, mean, color=colour, label=f"{label} (n={n})", solid_capstyle="round")
    if 1 < len(groups) <= MAX_DIRECT_LABELS:
        ends = sorted(groups.items(), key=lambda kv: kv[1][0][-1])
        gap = 0.075 * (top.get_ylim()[1] - top.get_ylim()[0])
        y_prev = -np.inf
        for label, (mean, _, _) in ends:  # nudge labels apart so they never collide
            y = max(mean[-1], y_prev + gap)
            top.annotate(
                label, (t[-1], y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2
            )
            y_prev = y
    _event_line(top)
    top.grid(axis="y")
    top.set_ylabel(rate_label)
    _title_and_legend(top, title, min(len(groups), 4) if len(groups) > 1 else 0)
    top.margins(x=0)
    if not heatmap:
        top.set_xlabel(f"Time from {event} (s)")
        return _save(fig, stem)

    order = np.argsort(np.abs(unit_z).argmax(axis=1))
    lim = max(2.0, float(np.percentile(np.abs(unit_z), 99)))
    step = t[1] - t[0]
    im = bottom.imshow(
        unit_z[order],
        aspect="auto",
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        extent=(t[0] - step / 2, t[-1] + step / 2, len(order), 0),
        interpolation="nearest",
    )
    _event_line(bottom)
    bottom.set_ylabel("Units, sorted by time of peak change")
    bottom.set_xlabel(f"Time from {event} (s)")
    bar = fig.colorbar(im, ax=[top, bottom], fraction=0.035, pad=0.09, shrink=0.5, anchor=(0, 0.1))
    bar.outline.set_visible(False)
    bar.set_label("Change from baseline (SD)", color=INK_2, fontsize=8)
    return _save(fig, stem)


def plot_raster(stem: str, title: str, spikes: list[np.ndarray], t0: float, t1: float, valid: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    edges = np.concatenate([[t0], np.clip(valid, t0, t1).ravel(), [t1]])
    for a, b in edges.reshape(-1, 2):  # spans between recorded intervals
        if b > a:
            ax.axvspan(a, b, color=UNRECORDED, linewidth=0, zorder=0)
    ax.eventplot(spikes, colors=INK, linewidths=0.6, linelengths=0.8, lineoffsets=np.arange(len(spikes)) + 0.5)
    ax.set_xlim(t0, t1)
    ax.set_ylim(len(spikes), 0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    ax.set_title(title)
    if (np.diff(edges.reshape(-1, 2), axis=1) > 0).any():
        ax.annotate(
            "shaded = not recorded",
            (1, 1.01),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    return _save(fig, stem)


def plot_decode(stem: str, title: str, t, true: np.ndarray, decoded: np.ndarray, unit: str, r2: np.ndarray) -> Path:
    dims = true.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(7.2, 1.6 + 1.9 * dims), sharex=True, squeeze=False)
    names = ["x", "y", "z"] if dims <= 3 else [str(i) for i in range(dims)]
    gaps = np.flatnonzero(np.diff(t) > 1.5 * np.median(np.diff(t))) + 1  # break lines across unrecorded spans
    for i, ax in enumerate(axes[:, 0]):
        for series, colour, label in (
            (true[:, i], CATEGORICAL[0], "Actual"),
            (decoded[:, i], CATEGORICAL[1], "Decoded"),
        ):
            ax.plot(np.insert(t, gaps, np.nan), np.insert(series, gaps, np.nan), color=colour, label=label)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
        ax.grid(axis="y")
        ax.margins(x=0)
        ax.set_ylabel(f"{names[i]} ({unit})" if unit else names[i])
        ax.annotate(
            f"R² {r2[i]:.2f}", (1, 1.03), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color=INK_2
        )
    _title_and_legend(axes[0, 0], title, 2)
    axes[-1, 0].set_xlabel("Time (s)")
    return _save(fig, stem)


def plot_latents(
    stem: str,
    title: str,
    t,
    traj: np.ndarray,
    labels: np.ndarray,
    variance: np.ndarray,
    max_trials: int,
    variance_label: str = "Variance explained (%)",
) -> Path:
    """traj: (n_trials, n_bins, n_factors); labels: group per trial."""
    groups = sorted(set(labels.tolist()))
    colour = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(groups)}
    n_show = min(3, traj.shape[2])
    fig = plt.figure(figsize=(9.6, 2.2 + 1.7 * n_show))
    grid = fig.add_gridspec(n_show, 2, width_ratios=[1.35, 1], hspace=0.28, wspace=0.28)
    rng = np.random.default_rng(0)
    drawn = rng.choice(len(labels), size=min(max_trials, len(labels)), replace=False)
    axes = [fig.add_subplot(grid[k, 0]) for k in range(n_show)]
    for k, ax in enumerate(axes):
        for i in drawn:
            ax.plot(t, traj[i, :, k], color=colour[labels[i]], alpha=0.18, linewidth=0.8)
        for g in groups:
            ax.plot(t, traj[labels == g, :, k].mean(axis=0), color=colour[g], linewidth=2.2, label=g)
        _event_line(ax)
        ax.grid(axis="y")
        ax.margins(x=0)
        ax.set_ylabel(f"Factor {k + 1}")
        if k < n_show - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time from event (s)")
    _title_and_legend(axes[0], title, min(len(groups), 4) if len(groups) > 1 else 0)

    if traj.shape[2] >= 2:
        state = fig.add_subplot(grid[: max(1, n_show - 1), 1])
        for i in drawn:
            state.plot(traj[i, :, 0], traj[i, :, 1], color=colour[labels[i]], alpha=0.15, linewidth=0.7)
        zero = int(np.argmin(np.abs(t)))
        for g in groups:
            m = traj[labels == g].mean(axis=0)
            state.plot(m[:, 0], m[:, 1], color=colour[g], linewidth=2.2)
            state.plot(m[0, 0], m[0, 1], "o", color=colour[g], markersize=5, markeredgecolor=SURFACE)
            state.plot(m[zero, 0], m[zero, 1], "s", color=colour[g], markersize=6, markeredgecolor=SURFACE)
            state.plot(m[-1, 0], m[-1, 1], "^", color=colour[g], markersize=6, markeredgecolor=SURFACE)
        state.set_xlabel("Factor 1")
        state.set_ylabel("Factor 2")
        state.set_title("State space · ○ start  ■ event  ▲ end", fontsize=9, fontweight="normal", color=INK_2)
        state.grid(True)
        state.set_aspect("equal", adjustable="datalim")

    bars = fig.add_subplot(grid[-1, 1])
    idx = np.arange(1, len(variance) + 1)
    bars.bar(idx, 100 * variance, color=CATEGORICAL[0], width=0.7)
    bars.set_xticks(idx)
    bars.set_xlabel("Factor")
    bars.set_ylabel(variance_label)
    bars.grid(axis="y")
    bars.margins(y=0.15)
    for x, v in zip(idx, variance):
        bars.annotate(
            f"{100 * v:.1f}",
            (x, 100 * v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=INK_2,
        )
    return _save(fig, stem)
