import datetime as dt

import numpy as np

from openenergy.assets.repository import Plant, create_plant
from openenergy.retraining.drift import (
    assess_nwp_covariate_drift,
    check_drift,
    drift_status,
    population_stability_index,
    recent_nrmse,
    two_sample_ks,
)
from openenergy.storage import connect, init_schema


def _plant(db, capacity=10.0):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=capacity))


def _champion(db, *, plant_id="wf1", horizon=24, nrmse=0.05):
    """Seed a champion experiment_trial carrying a reference test_nrmse."""
    db.execute(
        "INSERT INTO experiments (experiment_id, plant_id) VALUES (1, ?)", [plant_id]
    )
    db.execute(
        "INSERT INTO experiment_trials (experiment_id, horizon_hours, model_family, "
        "test_nrmse, is_champion) VALUES (1, ?, 'lightgbm', ?, true)",
        [horizon, nrmse],
    )


def _write_forecasts(db, *, n, err, rng, plant_id="wf1", horizon=24, point_id=7,
                     base_hour=0, capacity=10.0):
    """Insert n (forecast, actual) pairs where the stored p50 misses the realised
    production by roughly ``err`` MW of noise. Production is the ground truth."""
    base = dt.datetime(2024, 6, 1) + dt.timedelta(hours=base_hour)
    for i in range(n):
        vt = base + dt.timedelta(hours=i)
        actual = 4.0 + 2.0 * np.sin(i / 6.0)
        actual = float(min(max(actual, 0.0), capacity))
        fc = float(min(max(actual + rng.normal(0, err), 0.0), capacity))
        issue = vt - dt.timedelta(hours=horizon)
        db.execute(
            "INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
            [plant_id, vt, actual],
        )
        db.execute(
            "INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, "
            "valid_time, p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)",
            [plant_id, point_id, horizon, issue, vt, fc - 1.0, fc, fc + 1.0, None],
        )


def test_recent_nrmse_reflects_error(db):
    _plant(db)
    rng = np.random.default_rng(0)
    _write_forecasts(db, n=60, err=2.0, rng=rng)
    val, n = recent_nrmse(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0)
    assert n == 60
    # ~2 MW RMS error on a 10 MW plant → nRMSE around 0.2.
    assert 0.1 < val < 0.35


def test_recent_nrmse_insufficient_data(db):
    _plant(db)
    rng = np.random.default_rng(0)
    _write_forecasts(db, n=5, err=0.5, rng=rng)
    val, n = recent_nrmse(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                          min_samples=12)
    assert val is None
    assert n == 5


def test_healthy_forecasts_do_not_fire(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(1)
    # Recent error ~0.5 MW → nRMSE ~0.05, at champion skill: no breach.
    _write_forecasts(db, n=60, err=0.5, rng=rng)
    calls = []
    for _ in range(3):
        res = check_drift(
            db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
            min_consecutive=2, enqueue=lambda p, h: calls.append((p, h)) or 999,
        )
    assert calls == []
    assert res["status"] == "ok"
    assert res["fired"] is False


def test_drift_fires_once_hysteresis(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(2)
    # Degraded: ~3 MW RMS error → nRMSE ~0.3 ≫ 1.5 × 0.05 breach threshold.
    _write_forecasts(db, n=80, err=3.0, rng=rng)

    calls = []
    enqueue = lambda p, h: (calls.append((p, h)), 555)[1]

    r1 = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                     min_consecutive=2, enqueue=enqueue)
    # First breach only arms the counter (persistence), does not fire yet.
    assert r1["fired"] is False
    assert calls == []

    # New realised data → a SECOND distinct evaluation window (T-06): persistence is
    # over windows, not repeated calls on the same data.
    _write_forecasts(db, n=40, err=3.0, rng=rng, base_hour=80)
    r2 = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                     min_consecutive=2, enqueue=enqueue)
    # Second consecutive breaching window fires the emergency retrain exactly once.
    assert r2["fired"] is True
    assert r2["job_id"] == 555
    assert calls == [("wf1", 24)]

    r3 = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                     min_consecutive=2, enqueue=enqueue)
    # Same window, already fired → no second enqueue (hysteresis + window-gating).
    assert r3["fired"] is False
    assert r3["status"] == "armed"
    assert calls == [("wf1", 24)]


def test_recovery_rearms_after_fire(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(3)
    _write_forecasts(db, n=80, err=3.0, rng=rng)

    calls = []
    enqueue = lambda p, h: (calls.append((p, h)), 1)[1]
    check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                min_consecutive=1, enqueue=enqueue)
    assert len(calls) == 1  # fired

    # Replace forecasts with healthy ones (recovery).
    db.execute("DELETE FROM forecasts")
    db.execute("DELETE FROM production")
    _write_forecasts(db, n=80, err=0.4, rng=rng, base_hour=200)
    r_rec = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                        min_consecutive=1, enqueue=enqueue)
    assert r_rec["fired"] is False
    assert r_rec["status"] == "ok"
    assert len(calls) == 1  # no new fire during recovery

    # Degrade again → re-arms and fires a second time.
    db.execute("DELETE FROM forecasts")
    db.execute("DELETE FROM production")
    _write_forecasts(db, n=80, err=3.0, rng=rng, base_hour=400)
    check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                min_consecutive=1, enqueue=enqueue)
    assert len(calls) == 2


def test_no_champion_does_not_fire(db):
    _plant(db)
    rng = np.random.default_rng(4)
    _write_forecasts(db, n=80, err=3.0, rng=rng)
    calls = []
    res = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                      min_consecutive=1, enqueue=lambda p, h: calls.append(1) or 1)
    assert res["status"] == "no_champion"
    assert res["fired"] is False
    assert calls == []


def test_insufficient_data_does_not_fire(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(5)
    _write_forecasts(db, n=4, err=3.0, rng=rng)
    calls = []
    res = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                      min_consecutive=1, min_samples=12,
                      enqueue=lambda p, h: calls.append(1) or 1)
    assert res["status"] == "insufficient_data"
    assert res["fired"] is False
    assert calls == []


def test_drift_status_surface(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(6)
    _write_forecasts(db, n=80, err=3.0, rng=rng)
    check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                min_consecutive=1, enqueue=lambda p, h: 42)
    rows = drift_status(db, plant_id="wf1")
    assert len(rows) == 1
    row = rows[0]
    assert row["horizon_hours"] == 24
    assert row["breaching"] is True
    assert row["last_job_id"] == 42
    assert row["recent_nrmse"] > row["champion_nrmse"]


def test_old_schema_migrates_drift_state_table(tmp_path):
    db_file = tmp_path / "old.duckdb"
    con = connect(db_file)
    init_schema(con)
    con.execute("DROP TABLE drift_state")  # simulate a pre-E1-t4 DB
    con.close()

    con2 = connect(db_file)
    init_schema(con2)  # must recreate the table idempotently
    con2.execute(
        "INSERT INTO drift_state (plant_id, horizon_hours, consecutive_breaches, "
        "fired) VALUES ('p', 24, 1, true)"
    )
    n = con2.execute("SELECT count(*) FROM drift_state").fetchone()[0]
    assert n == 1
    con2.close()


def test_reference_follows_serving_champion(db):
    """T-06: drift must reference the SERVED strategy champion (champion_pointer →
    strategy_trials), not the offline experiment_trials champion. With both present
    at different skill, the served (0.05) is used, not the offline one (0.30)."""
    _plant(db)
    db.execute("INSERT INTO experiments (experiment_id, plant_id) VALUES (1, 'wf1')")
    db.execute(
        "INSERT INTO experiment_trials (experiment_id, horizon_hours, model_family, "
        "test_nrmse, is_champion) VALUES (1, 24, 'lightgbm', 0.30, true)"
    )
    db.execute(
        "INSERT INTO strategy_trials (experiment_id, horizon_hours, ensemble_method, "
        "strategy_config, test_nrmse) VALUES (1, 24, 'mean', '{}', 0.05)"
    )
    stid = db.execute(
        "SELECT strategy_trial_id FROM strategy_trials WHERE experiment_id=1 AND horizon_hours=24"
    ).fetchone()[0]
    db.execute(
        "INSERT INTO champion_pointer (plant_id, horizon_hours, strategy_trial_id) "
        "VALUES ('wf1', 24, ?)", [stid]
    )
    rng = np.random.default_rng(0)
    _write_forecasts(db, n=60, err=2.0, rng=rng)
    res = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                      min_consecutive=2, enqueue=lambda p, h: 1)
    assert res["champion_nrmse"] == 0.05


def test_repeat_call_same_window_never_fires(db):
    """T-06: the persistence guarantee is over evaluation WINDOWS. Re-running
    check_drift on unchanged realised data (same window) must never advance the
    counter or enqueue a retrain — a loop of no-op calls cannot trip a retrain."""
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(11)
    _write_forecasts(db, n=80, err=3.0, rng=rng)  # genuinely degraded
    calls = []
    enqueue = lambda p, h: (calls.append((p, h)), 1)[1]
    last = None
    for _ in range(10):
        last = check_drift(db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
                           min_consecutive=2, enqueue=enqueue)
    assert calls == []  # no new realised data → never fires despite 10 calls
    assert last["fired"] is False
    assert last["status"] == "degraded"


def test_psi_identical_near_zero():
    values = np.r_[np.linspace(-2.0, 2.0, 200), np.nan]
    assert population_stability_index(values, values.copy()) < 1e-10
    # Constants are a supported, meaningful edge case rather than a binning error.
    assert population_stability_index(np.ones(20), np.ones(20)) < 1e-10


def test_psi_shifted_exceeds_threshold():
    rng = np.random.default_rng(33)
    reference = rng.normal(0.0, 1.0, 500)
    current = rng.normal(2.0, 1.0, 500)
    assert population_stability_index(reference, current) > 0.25
    statistic, pvalue = two_sample_ks(reference, current)
    assert statistic > 0.5
    assert pvalue < 0.05


def test_stale_data_does_not_report_ok(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(34)
    _write_forecasts(db, n=60, err=0.4, rng=rng)
    stale_issue = dt.datetime(2024, 1, 1)
    for i in range(40):
        valid = dt.datetime(2024, 1, 2) + dt.timedelta(hours=i)
        db.execute(
            "INSERT INTO weather_raw "
            "(point_id, role, model, valid_time, issue_time, lead_hours, variable, value) "
            "VALUES (7, 'previous_runs', 'icon', ?, ?, 24, 'temperature_2m', ?)",
            [valid, stale_issue, float(i)],
        )

    result = check_drift(
        db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
        point_id=7, nwp_sources=["icon"], min_consecutive=2,
        as_of=dt.datetime(2024, 2, 1),
    )
    assert result["status"] != "ok"
    assert result["nwp_status"] == "nwp_stale"
    events = db.execute(
        "SELECT feature_name, detector_status FROM drift_events "
        "WHERE plant_id='wf1' AND horizon_hours=24"
    ).fetchall()
    assert ("__freshness__", "stale") in events


def test_missing_required_nwp_is_not_ok_and_does_not_enqueue(db):
    _plant(db)
    _champion(db, nrmse=0.05)
    rng = np.random.default_rng(35)
    _write_forecasts(db, n=60, err=0.4, rng=rng)
    calls = []
    result = check_drift(
        db, plant_id="wf1", horizon_hours=24, capacity_mw=10.0,
        point_id=7, nwp_sources=["icon"], min_consecutive=1,
        enqueue=lambda p, h: calls.append((p, h)) or 1,
    )
    assert result["status"] == "nwp_missing"
    assert calls == []


def test_covariate_results_are_persisted_per_feature(db):
    _plant(db)
    start = dt.datetime(2024, 1, 1)
    for i in range(120):
        valid = start + dt.timedelta(days=i)
        value = float(i % 7)
        db.execute(
            "INSERT INTO weather_raw "
            "(point_id, role, model, valid_time, issue_time, lead_hours, variable, value) "
            "VALUES (7, 'previous_runs', 'icon', ?, ?, 24, 'temperature_2m', ?)",
            [valid, dt.datetime(2024, 5, 1), value],
        )
    result = assess_nwp_covariate_drift(
        db, plant_id="wf1", point_id=7, horizon_hours=24, sources=["icon"],
        lookback_days=30, min_samples=10, as_of=dt.datetime(2024, 5, 2),
    )
    assert result["features"]
    row = db.execute(
        "SELECT psi_threshold, ks_alpha, reference_samples, current_samples, "
        "reference_season, metadata FROM drift_events "
        "WHERE feature_name='temperature_2m'"
    ).fetchone()
    assert row[0] == 0.25
    assert row[1] == 0.05
    assert row[2] >= 10 and row[3] >= 10
    assert row[4] is not None
    assert "reference_kind" in row[5]
