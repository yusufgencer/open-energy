from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainPolicy:
    name: str = "expanding_all"


_ROLLING_DAYS = {
    "rolling_90d": 90,
    "rolling_180d": 180,
    "rolling_365d": 365,
}

POLICY_NAMES = ["expanding_all", "rolling_90d", "rolling_180d", "rolling_365d", "recent_weighted"]

# Solar-only extra: weight training rows ∝ P_cs (clear-sky power) so scarce
# high-irradiance peak hours count more. Wind never sees this axis.
SOLAR_POLICY_NAMES = POLICY_NAMES + ["pcs_weighted"]

_VALID_POLICIES = set(SOLAR_POLICY_NAMES)


def select_train_indices(
    valid_time: np.ndarray,
    train_idx: np.ndarray,
    policy: str,
    *,
    min_rows: int = 20,
) -> np.ndarray:
    """Apply a train-window policy to an already leakage-safe train index set."""
    if policy not in _VALID_POLICIES:
        raise ValueError(f"bilinmeyen train_policy: {policy}")
    if policy not in _ROLLING_DAYS:
        return train_idx
    if len(train_idx) < min_rows:
        return train_idx

    vt = valid_time[train_idx].astype("datetime64[us]")
    cutoff = vt.max() - np.timedelta64(_ROLLING_DAYS[policy], "D")
    selected = train_idx[vt >= cutoff]
    return selected if len(selected) >= min_rows else train_idx


def sample_weights(
    valid_time: np.ndarray,
    train_idx: np.ndarray,
    policy: str,
    *,
    pcs: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return optional row weights for policies that reweight training rows.

    ``recent_weighted`` prefers recent observations; ``pcs_weighted`` weights
    rows ∝ P_cs (clear-sky power) so scarce high-irradiance peak hours count
    more, with a strictly positive floor so night rows are not fully dropped.
    Returns ``None`` (uniform weights) for every other policy, and for
    ``pcs_weighted`` when no P_cs series is supplied.
    """
    if len(train_idx) == 0:
        return None
    if policy == "pcs_weighted":
        if pcs is None:
            return None
        vals = np.asarray(pcs, dtype=float)[train_idx]
        vals = np.where(np.isfinite(vals), vals, 0.0)
        mx = float(vals.max()) if vals.size else 0.0
        floor = mx * 1e-3 if mx > 0.0 else 1.0
        return np.maximum(vals, floor)
    if policy != "recent_weighted":
        return None
    vt = valid_time[train_idx].astype("datetime64[us]").astype("int64")
    span = vt.max() - vt.min()
    if span <= 0:
        return np.ones(len(train_idx), dtype=float)
    recency = (vt - vt.min()) / span
    return 0.5 + recency
