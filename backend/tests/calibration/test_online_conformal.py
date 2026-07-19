import datetime as dt

import numpy as np

from openenergy.calibration.online import OnlineDtACI, refresh_and_apply
from openenergy.models.base import Prediction


def test_online_coverage_tracks_nominal_under_shift():
    rng = np.random.default_rng(321)
    n = 1600
    shift_at = n // 2
    y = rng.normal(np.where(np.arange(n) < shift_at, 0.0, 3.0), 1.0)
    base_lo = np.full(n, -0.5)
    base_hi = np.full(n, 0.5)
    state = OnlineDtACI(gamma=0.02, score_window=256, min_history=8)
    covered = np.zeros(n, dtype=bool)
    for i in range(n):
        pred = state.adjust(
            Prediction(
                p50=np.array([0.0]),
                p10=np.array([base_lo[i]]),
                p90=np.array([base_hi[i]]),
            )
        )
        lo, hi = float(pred.p10[0]), float(pred.p90[0])
        covered[i] = lo <= y[i] <= hi
        state.update(
            actual=float(y[i]),
            base_lower=float(base_lo[i]),
            base_upper=float(base_hi[i]),
            served_lower=lo,
            served_upper=hi,
        )

    tail_coverage = float(covered[shift_at + 300 :].mean())
    assert 0.72 <= tail_coverage <= 0.88


def test_feedback_is_past_only_and_same_valid_time_is_idempotent(db):
    issue = dt.datetime(2025, 1, 2)
    past = issue - dt.timedelta(hours=1)
    future = issue + dt.timedelta(hours=1)
    db.execute(
        "INSERT INTO plants (plant_id, name, kind, capacity_mw) VALUES ('p','P','wind',10)"
    )
    for valid_time, actual in [(past, 5.0), (future, 8.0)]:
        db.execute(
            "INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
            ["p", valid_time, actual],
        )
        db.execute(
            """INSERT INTO forecasts
               (plant_id, point_id, horizon_hours, issue_time, valid_time, p10, p50, p90)
               VALUES (?,?,?,?,?,?,?,?)""",
            ["p", 1, 24, valid_time - dt.timedelta(hours=24), valid_time, 0.0, 1.0, 2.0],
        )
    pred = Prediction(
        p50=np.array([1.0]), p10=np.array([0.0]), p90=np.array([2.0])
    )
    _out, version1 = refresh_and_apply(
        db, plant_id="p", horizon_hours=24, issue_time=issue, pred=pred
    )
    _out, version2 = refresh_and_apply(
        db, plant_id="p", horizon_hours=24, issue_time=issue, pred=pred
    )
    assert version1 == version2 == 1
    rows = db.execute(
        "SELECT valid_time, state_version FROM online_conformal_feedback"
    ).fetchall()
    assert rows == [(past, 1)]
