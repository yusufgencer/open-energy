from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import polars as pl

from openenergy.datasets.horizon import (
    build_horizon_weather,
    build_multipoint_horizon_weather,
)
from openenergy.features import FeatureConfig, build_features
from openenergy.features.point_policy import apply_point_policy, is_multipoint


def forecast_matrix(con: duckdb.DuckDBPyConnection, *, point_id: int, kind: str,
                    feature_blocks: list[str], feature_names: list[str],
                    issue_time: dt.datetime, horizon_hours: int,
                    source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build serving feature matrix aligned to training feature_names.

    Returns (X, valid_times):
    - X: numpy array of shape (n, len(feature_names)) — columns in feature_names order
    - valid_times: numpy array of valid_time values for selected rows

    Missing columns → zeros; extra columns dropped. Uses the exact training DGP:
    role=PREVIOUS_RUNS, lead_hours=horizon_hours, and the shared horizon pivot.
    Only valid_time == issue_time + horizon_hours is selected for serving.
    """
    target_vt = issue_time + dt.timedelta(hours=horizon_hours)
    wide = build_horizon_weather(
        con, point_id=point_id, horizon_hours=horizon_hours,
        source=source, valid_time=target_vt,
    )
    if wide.height == 0:
        return np.empty((0, len(feature_names))), np.empty((0,), dtype="datetime64[us]")
    feats = build_features(wide, kind, FeatureConfig(blocks=feature_blocks))
    cols = []
    for name in feature_names:
        if name in feats.columns:
            cols.append(feats[name].cast(pl.Float64).to_numpy())
        else:
            cols.append(np.zeros(feats.height))
    X = np.column_stack(cols) if cols else np.empty((feats.height, 0))
    return X, feats["valid_time"].to_numpy()


def _forecast_multipoint_wide(con: duckdb.DuckDBPyConnection, *, point_ids: list[int],
                              issue_time: dt.datetime, horizon_hours: int,
                              source_policy: str | None) -> tuple[pl.DataFrame | None, set[int]]:
    """Serving slice of the shared previous-runs multipoint weather frame.

    Uses the training assembly's column layout exactly: each point's variables
    become ``{var}__p{pid}`` columns (``{var}__p{pid}__{src}`` for a combo source
    policy); points are inner-joined on valid_time so a missing grid point drops the
    row rather than silently substituting. The shared builder selects the same
    previous-runs lead as training and the one valid_time matching issue+horizon.

    Returns ``(merged_frame, contributing_point_ids)`` so the caller can enforce the
    explicit-degradation policy: a point that produced no target row is reported as
    absent rather than being silently averaged away.
    """
    target_vt = issue_time + dt.timedelta(hours=horizon_hours)
    return build_multipoint_horizon_weather(
        con, point_ids=point_ids, horizon_hours=horizon_hours,
        source_policy=source_policy, valid_time=target_vt,
    )


def forecast_matrix_multipoint(con: duckdb.DuckDBPyConnection, *, point_ids: list[int],
                               kind: str, feature_blocks: list[str], feature_names: list[str],
                               issue_time: dt.datetime, horizon_hours: int,
                               point_policy: str,
                               source_policy: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Serving matrix for a grid champion — rebuilds the exact training policy transform.

    Assembles the previous-runs grid frame (build_multipoint_dataset's layout),
    collapses it with the SAME apply_point_policy + point_ids as training, then
    featurizes with the stored feature_blocks and aligns to feature_names. The
    result is byte-for-byte the training-path featurization of the same weather.

    Missing grid points degrade explicitly: if any requested point_id has no
    previous-runs row at the target valid_time, the whole matrix is empty (the caller
    must not serve a vector collapsed over a partial grid), rather than silently
    averaging over whatever points happen to be present.
    """
    wide, present = _forecast_multipoint_wide(
        con, point_ids=point_ids, issue_time=issue_time,
        horizon_hours=horizon_hours, source_policy=source_policy,
    )
    if wide is None or wide.height == 0 or present != set(point_ids):
        return np.empty((0, len(feature_names))), np.empty((0,), dtype="datetime64[us]")
    wide = wide.sort("valid_time")
    if point_policy != "single_point" and is_multipoint(wide):
        wide = apply_point_policy(wide, point_policy, point_ids=point_ids)
    feats = build_features(wide, kind, FeatureConfig(blocks=feature_blocks))
    feats = feats.sort("valid_time")
    cols = []
    for name in feature_names:
        if name in feats.columns:
            cols.append(feats[name].cast(pl.Float64).to_numpy())
        else:
            cols.append(np.zeros(feats.height))
    X = np.column_stack(cols) if cols else np.empty((feats.height, 0))
    return X, feats["valid_time"].to_numpy()
