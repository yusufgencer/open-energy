import datetime as dt

import numpy as np

from openenergy.forecasting.reforecast import blend_bands, blend_stored_run, reforecast
from openenergy.models.base import Prediction


def test_blend_bands_weighted_average_and_monotone():
    new = Prediction(p50=np.array([10.0]), p10=np.array([8.0]), p90=np.array([12.0]))
    old = Prediction(p50=np.array([20.0]), p10=np.array([18.0]), p90=np.array([22.0]))
    b = blend_bands(new, old, w_new=0.7)
    assert abs(b.p50[0] - 13.0) < 1e-9   # 0.7*10 + 0.3*20
    assert abs(b.p10[0] - 11.0) < 1e-9
    assert abs(b.p90[0] - 15.0) < 1e-9
    assert b.p10[0] <= b.p50[0] <= b.p90[0]


def _ins(db, plant_id, horizon, issue, vt, p10, p50, p90):
    db.execute(
        """INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, valid_time,
           p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)""",
        [plant_id, 1, horizon, issue, vt, p10, p50, p90, None],
    )


def test_blend_matches_across_horizons_by_valid_time(db):
    """T-03: the time-lagged blend must match runs that target the SAME valid_time
    from DIFFERENT lead times (issue+horizon). Matching within one horizon can
    never overlap: valid_time = issue + horizon, so two issues have disjoint
    valid_times. Here the fresh H=24 run (issue=V-24) is blended toward the prior
    H=48 run (issue=V-48) that already forecast the same valid_time V."""
    V = dt.datetime(2026, 1, 3, 0)
    old_issue = V - dt.timedelta(hours=48)   # H=48 → valid_time = V
    new_issue = V - dt.timedelta(hours=24)   # H=24 → valid_time = V
    _ins(db, "p", 48, old_issue, V, 18.0, 20.0, 22.0)   # prior run, longer lead
    _ins(db, "p", 24, new_issue, V, 8.0, 10.0, 12.0)    # fresh run, shorter lead
    updated = blend_stored_run(db, plant_id="p", horizon_hours=24,
                               new_issue=new_issue, w_new=0.7)
    assert updated == 1
    row = db.execute(
        "SELECT p10, p50, p90 FROM forecasts "
        "WHERE plant_id='p' AND issue_time=? AND horizon_hours=24", [new_issue]
    ).fetchone()
    assert abs(row[1] - 13.0) < 1e-9   # 0.7*10 + 0.3*20
    assert abs(row[0] - 11.0) < 1e-9 and abs(row[2] - 15.0) < 1e-9


def test_blend_no_prior_valid_time_is_noop(db):
    """T-03 honest contract: with no earlier run targeting this valid_time, the
    blend updates nothing (updated == 0) — the caller must then report blended=False."""
    V = dt.datetime(2026, 1, 3, 0)
    new_issue = V - dt.timedelta(hours=24)
    _ins(db, "p", 24, new_issue, V, 8.0, 10.0, 12.0)  # only the fresh run exists
    updated = blend_stored_run(db, plant_id="p", horizon_hours=24,
                               new_issue=new_issue, w_new=0.7)
    assert updated == 0


def test_reforecast_reports_blended_only_when_a_row_was_updated(db, monkeypatch):
    """T-03: the public result must reflect an actual valid-time match."""
    V = dt.datetime(2026, 1, 3, 0)
    old_issue = V - dt.timedelta(hours=48)
    new_issue = V - dt.timedelta(hours=24)
    _ins(db, "p", 48, old_issue, V, 18.0, 20.0, 22.0)

    def fake_generate(con, **kwargs):
        _ins(con, kwargs["plant_id"], kwargs["horizon_hours"], kwargs["issue_time"],
             V, 8.0, 10.0, 12.0)
        return 1

    monkeypatch.setattr(
        "openenergy.forecasting.reforecast.generate_forecast", fake_generate
    )
    result = reforecast(
        db, plant_id="p", point_id=1, horizon_hours=24,
        capacity_mw=10.0, kind="wind", issue_time=new_issue, w_new=0.7,
    )

    assert result == {"fired": True, "blended": True, "rows": 1}
    p50 = db.execute(
        "SELECT p50 FROM forecasts "
        "WHERE plant_id='p' AND horizon_hours=24 AND issue_time=?",
        [new_issue],
    ).fetchone()[0]
    assert abs(p50 - 13.0) < 1e-9


def test_reforecast_reports_not_blended_without_valid_time_match(db, monkeypatch):
    """T-03: an unrelated prior run must not produce a false blended=True."""
    V = dt.datetime(2026, 1, 3, 0)
    new_issue = V - dt.timedelta(hours=24)
    unrelated_vt = V + dt.timedelta(hours=1)
    _ins(db, "p", 49, V - dt.timedelta(hours=49), unrelated_vt,
         18.0, 20.0, 22.0)

    def fake_generate(con, **kwargs):
        _ins(con, kwargs["plant_id"], kwargs["horizon_hours"], kwargs["issue_time"],
             V, 8.0, 10.0, 12.0)
        return 1

    monkeypatch.setattr(
        "openenergy.forecasting.reforecast.generate_forecast", fake_generate
    )
    result = reforecast(
        db, plant_id="p", point_id=1, horizon_hours=24,
        capacity_mw=10.0, kind="wind", issue_time=new_issue, w_new=0.7,
    )

    assert result == {"fired": True, "blended": False, "rows": 1}
    p50 = db.execute(
        "SELECT p50 FROM forecasts "
        "WHERE plant_id='p' AND horizon_hours=24 AND issue_time=?",
        [new_issue],
    ).fetchone()[0]
    assert p50 == 10.0


def test_reforecast_fires_once_per_init_time(db):
    t1 = dt.datetime(2026, 1, 1, 3)
    vt = dt.datetime(2026, 1, 2, 0)
    _ins(db, "p", 24, t1, vt, 8.0, 10.0, 12.0)  # this init_time already produced forecasts
    res = reforecast(db, plant_id="p", point_id=1, horizon_hours=24,
                     capacity_mw=10.0, kind="wind", issue_time=t1)
    assert res["fired"] is False and res["rows"] == 0
