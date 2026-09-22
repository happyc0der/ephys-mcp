"""Contact layouts for common arrays, in micrometres, and a summary of any layout."""

from __future__ import annotations

import numpy as np

LAYOUTS = ("linear", "grid", "utah", "tetrode_linear")


def make_layout(layout: str, n_channels: int, pitch_um: float, n_columns: int = 1) -> np.ndarray:
    """Positions (n_channels, 2) for a named layout, channel order row-major from the origin."""
    if pitch_um <= 0:
        raise ValueError("pitch_um must be positive")
    if layout == "linear":
        return np.c_[np.zeros(n_channels), pitch_um * np.arange(n_channels)]
    if layout in ("grid", "utah"):
        cols = int(np.ceil(np.sqrt(n_channels))) if layout == "utah" or n_columns < 1 else n_columns
        idx = np.arange(n_channels)
        return np.c_[pitch_um * (idx % cols), pitch_um * (idx // cols)].astype(float)
    if layout == "tetrode_linear":  # groups of four contacts 25 um apart, groups pitch_um apart
        idx = np.arange(n_channels)
        local = np.array([[0, 0], [25, 0], [0, 25], [25, 25]], float)
        return local[idx % 4] + np.c_[np.zeros(n_channels), pitch_um * (idx // 4)]
    raise ValueError(f"layout must be one of {LAYOUTS}")


def summarize(positions: np.ndarray) -> dict:
    span = positions.max(axis=0) - positions.min(axis=0)
    # nearest-neighbour distance, the practical "pitch"
    if len(positions) > 1:
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        pitch = float(np.median(dist.min(axis=1)))
    else:
        pitch = 0.0
    return {
        "n_contacts": len(positions),
        "span_um": [round(float(v), 1) for v in span],
        "nearest_neighbour_um": round(pitch, 1),
        "dense": bool(pitch and pitch < 100),
        "extent_note": "contacts under 100 um apart see the same units; sorting treats them jointly"
        if pitch and pitch < 100
        else "contacts are far apart; each mostly sees its own units",
    }
