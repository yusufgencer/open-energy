"""Dynamic reforecasting with time-lagged run blending (F1-t4).

Operational renewable forecasting reforecasts whenever a new NWP run arrives
(not on a wall-clock schedule). To cut the forecast "jumpiness" traders dislike,
the freshest run is blended with the previous one (weighted toward the newest,
~0.7/0.3) — a time-lagged ensemble. reforecast() fires at most once per NWP
init time (issue_time): if forecasts already exist for that issue_time it is a
no-op, so re-triggering the same run is safe.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np

from openenergy.forecasting.predict import generate_forecast
from openenergy.models.base import Prediction, enforce_quantile_order


def blend_bands(new: Prediction, old: Prediction, w_new: float = 0.7) -> Prediction:
    """Time-lagged blend of two forecasts: w_new·new + (1-w_new)·old per quantile.

    p50 is always blended; a band is blended only when both runs provide it,
    otherwise the newer run's band passes through. The result is re-ordered so
    p10 ≤ p50 ≤ p90 holds after blending.
    """
    w_old = 1.0 - w_new

    def blend(a, b):
        if a is None or b is None:
            return a
        return w_new * np.asarray(a, dtype=float) + w_old * np.asarray(b, dtype=float)

    return enforce_quantile_order(Prediction(
        p50=w_new * np.asarray(new.p50, dtype=float) + w_old * np.asarray(old.p50, dtype=float),
        p10=blend(new.p10, old.p10),
        p90=blend(new.p90, old.p90),
    ))


def blend_stored_run(con: duckdb.DuckDBPyConnection, *, plant_id: str, horizon_hours: int,
                     new_issue: dt.datetime, w_new: float = 0.7) -> int:
    """Blend the freshly-written new-run forecast rows in place with the prior run.

    A time-lagged ensemble blends runs that target the SAME valid_time from
    DIFFERENT lead times. Because ``valid_time == issue_time + horizon``, two
    runs at the same horizon have disjoint valid_times, so the prior counterpart
    for a fresh row at valid_time V is the most recently issued EARLIER forecast
    for the same V — at whatever (longer) horizon produced it. A fresh hour with
    no earlier counterpart is left as-is. Returns the number of rows updated.
    """
    new_rows = con.execute(
        "SELECT valid_time, p10, p50, p90 FROM forecasts "
        "WHERE plant_id=? AND horizon_hours=? AND issue_time=?",
        [plant_id, horizon_hours, new_issue],
    ).fetchall()
    w_old = 1.0 - w_new
    updated = 0
    for vt, p10, p50, p90 in new_rows:
        prior = con.execute(
            "SELECT p10, p50, p90 FROM forecasts "
            "WHERE plant_id=? AND valid_time=? AND issue_time < ? "
            "ORDER BY issue_time DESC LIMIT 1",
            [plant_id, vt, new_issue],
        ).fetchone()
        if prior is None:
            continue
        o10, o50, o90 = prior
        b50 = w_new * p50 + w_old * o50
        b10 = w_new * p10 + w_old * o10 if (p10 is not None and o10 is not None) else p10
        b90 = w_new * p90 + w_old * o90 if (p90 is not None and o90 is not None) else p90
        if b10 is not None:
            b10 = min(b10, b50)  # keep p50 as the anchor (matches enforce_quantile_order)
        if b90 is not None:
            b90 = max(b90, b50)
        con.execute(
            "UPDATE forecasts SET p10=?, p50=?, p90=? "
            "WHERE plant_id=? AND horizon_hours=? AND issue_time=? AND valid_time=?",
            [b10, b50, b90, plant_id, horizon_hours, new_issue, vt],
        )
        updated += 1
    return updated


def reforecast(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
               horizon_hours: int, capacity_mw: float, kind: str,
               issue_time: dt.datetime, w_new: float = 0.7) -> dict:
    """Reforecast on arrival of a new NWP run, blending with the previous run.

    Fires at most once per issue_time (NWP init): if forecasts for this
    issue_time already exist it is a no-op. Otherwise it generates a fresh
    champion forecast and, when a prior run exists, blends the new rows toward
    it (w_new/1-w_new) to smooth run-to-run jumps.

    Returns {"fired": bool, "blended": bool, "rows": int}.
    """
    existing = con.execute(
        "SELECT count(*) FROM forecasts WHERE plant_id=? AND horizon_hours=? AND issue_time=?",
        [plant_id, horizon_hours, issue_time],
    ).fetchone()[0]
    if existing:
        return {"fired": False, "blended": False, "rows": 0}

    rows = generate_forecast(con, plant_id=plant_id, point_id=point_id,
                             horizon_hours=horizon_hours, capacity_mw=capacity_mw,
                             kind=kind, issue_time=issue_time)
    updated = 0
    if rows:
        updated = blend_stored_run(con, plant_id=plant_id, horizon_hours=horizon_hours,
                                   new_issue=issue_time, w_new=w_new)
    # blended is honest: True only when a prior run actually blended into a row.
    return {"fired": True, "blended": updated > 0, "rows": rows}
