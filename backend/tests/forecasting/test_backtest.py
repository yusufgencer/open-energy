import datetime as dt

import numpy as np
import polars as pl

from openenergy.assets.repository import Plant, create_plant
from openenergy.experiments.runner import run_experiment
from openenergy.forecasting.backtest import backfill_forecasts, backtest_champion
from openenergy.ingestion.production import write_production
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.retraining.drift import recent_nrmse
from openenergy.retraining.postprocessor import fit_postprocessor


def _populate_grid(db, plant_id="bt", n=240):
    create_plant(db, Plant(plant_id=plant_id, name="BT", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(0)
    ws = 5 + 3 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, n)
    for pid, mult in [(1, 1.0), (2, 1.05)]:
        rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
        for i, vt in enumerate(vts):
            for var, val in [("wind_speed_100m", ws[i] * mult), ("wind_speed_10m", ws[i] * 0.6),
                             ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
                rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
                rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
        frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                     "lead_hours": pl.Int32, "value": pl.Float64})
        write_weather_series(db, WeatherSeries(point_id=pid, role=ApiRole.PREVIOUS_RUNS.value,
                                               model="best_match", frame=frame))
    power = np.clip((ws ** 3) / 1000.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)


def test_backtest_champion_returns_test_window_series(db, tmp_path):
    _populate_grid(db)
    run_experiment(db, plant_id="bt", point_id=1, horizon_hours_list=[24], capacity_mw=10.0,
                   kind="wind", nwp_sources=["icon"], n_trials=5, n_splits=3, embargo=24,
                   seed=42, models_dir=tmp_path, point_ids=[1, 2])

    series = backtest_champion(db, plant_id="bt", horizon_hours=24, capacity_mw=10.0, kind="wind")
    assert len(series) > 0
    for row in series:
        assert {"valid_time", "actual", "p50"} <= set(row)
        assert 0.0 <= row["actual"] <= 10.0 + 1e-6   # within capacity
        assert 0.0 <= row["p50"] <= 10.0 + 1e-6
    # the point forecast tracks reality: correlation with actual is clearly positive
    a = np.array([r["actual"] for r in series])
    p = np.array([r["p50"] for r in series])
    if a.std() > 0 and p.std() > 0:
        assert np.corrcoef(a, p)[0, 1] > 0.3


def test_backtest_champion_no_champion_returns_empty(db):
    create_plant(db, Plant(plant_id="empty", name="E", kind="wind", capacity_mw=10))
    assert backtest_champion(db, plant_id="empty", horizon_hours=24, capacity_mw=10.0, kind="wind") == []


def _trained_backfill_fixture(db, tmp_path):
    _populate_grid(db)
    run_experiment(
        db, plant_id="bt", point_id=1, horizon_hours_list=[24], capacity_mw=10.0,
        kind="wind", nwp_sources=["icon"], n_trials=3, n_splits=3, embargo=24,
        seed=42, models_dir=tmp_path, point_ids=[1, 2],
    )
    return {
        "plant_id": "bt",
        "point_id": 1,
        "horizons": [24],
        "capacity_mw": 10.0,
        "kind": "wind",
        "start_time": dt.datetime(2024, 1, 9),
        "end_time": dt.datetime(2024, 1, 9, 23),
    }


def test_backfill_is_idempotent(db, tmp_path):
    kwargs = _trained_backfill_fixture(db, tmp_path)
    # A genuine forward-serving row may already occupy one historical natural
    # key by the time the backfill runs. It is authoritative and must not be
    # rewritten by the hindcast.
    existing_valid = kwargs["start_time"]
    existing_issue = existing_valid - dt.timedelta(hours=24)
    pointer_id = db.execute(
        "SELECT strategy_trial_id FROM champion_pointer "
        "WHERE plant_id = 'bt' AND horizon_hours = 24"
    ).fetchone()[0]
    db.execute(
        """
        INSERT INTO forecasts
            (plant_id, point_id, horizon_hours, issue_time, valid_time,
             p10, p50, p90, strategy_trial_id)
        VALUES ('bt', 1, 24, ?, ?, 122.0, 123.0, 124.0, ?)
        """,
        [existing_issue, existing_valid, pointer_id],
    )
    first = backfill_forecasts(db, **kwargs)
    assert first["rows"] == 23
    assert first["skipped"] == 1

    before = db.execute(
        """
        SELECT issue_time, valid_time, p50, strategy_trial_id, created_at
        FROM forecasts
        WHERE plant_id = 'bt' AND point_id = 1 AND horizon_hours = 24
        ORDER BY valid_time
        """
    ).fetchall()
    second = backfill_forecasts(db, **kwargs)
    after = db.execute(
        """
        SELECT issue_time, valid_time, p50, strategy_trial_id, created_at
        FROM forecasts
        WHERE plant_id = 'bt' AND point_id = 1 AND horizon_hours = 24
        ORDER BY valid_time
        """
    ).fetchall()

    assert second["rows"] == 0
    assert second["skipped"] == 24
    assert after == before
    assert len(after) == 24
    assert all(issue == valid - dt.timedelta(hours=24) for issue, valid, *_ in after)
    assert all(strategy_trial_id is not None for *_, strategy_trial_id, _ in after)
    assert db.execute(
        "SELECT p50 FROM forecasts WHERE plant_id = 'bt' AND valid_time = ?",
        [existing_valid],
    ).fetchone()[0] == 123.0


def test_backfill_populates_forecast_actual_pairs(db, tmp_path):
    kwargs = _trained_backfill_fixture(db, tmp_path)
    result = backfill_forecasts(db, **kwargs)

    pairs = db.execute(
        """
        SELECT count(*)
        FROM forecasts f
        JOIN production p
          ON p.plant_id = f.plant_id AND p.ts = f.valid_time
        WHERE f.plant_id = 'bt' AND f.point_id = 1 AND f.horizon_hours = 24
        """
    ).fetchone()[0]
    pointer_id = db.execute(
        """
        SELECT strategy_trial_id
        FROM champion_pointer
        WHERE plant_id = 'bt' AND horizon_hours = 24
        """
    ).fetchone()[0]
    lineage = db.execute(
        "SELECT DISTINCT strategy_trial_id FROM forecasts WHERE plant_id = 'bt'"
    ).fetchall()

    assert result["actual_pairs"] == 24
    assert pairs == 24
    assert lineage == [(pointer_id,)]
    # The two production consumers that motivated the backfill can immediately
    # read the newly created forecast/actual pairs.
    _nrmse, drift_samples = recent_nrmse(
        db, plant_id="bt", horizon_hours=24, capacity_mw=10.0, min_samples=1
    )
    postprocessor = fit_postprocessor(
        db, plant_id="bt", horizon_hours=24, min_samples=12
    )
    assert drift_samples == 24
    assert postprocessor["fitted"] is True
    assert postprocessor["n_samples"] == 24
