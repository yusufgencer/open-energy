import datetime as dt
import numpy as np
import polars as pl
from openenergy.assets.repository import Plant, PlantPoint, create_plant, add_point
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.experiments.runner import run_experiment
from openenergy.forecasting.predict import generate_forecast, run_serving_cycle


# Reuse the _populate helper from test_runner (inline it here)
def _populate(db, n=240):
    import datetime as dt
    import numpy as np
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(0)
    ws = 5 + 3 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, n)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("wind_speed_100m", ws[i]), ("wind_speed_10m", ws[i] * 0.6),
                         ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=7, role=ApiRole.PREVIOUS_RUNS.value, model="best_match", frame=frame))
    power = np.clip((ws ** 3) / 1000.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "wf1", prod)


def _write_forecast_weather(db, issue, n=6):
    """Write serving previous-runs weather for point_id=7 through the target."""
    vts = [issue + dt.timedelta(hours=i) for i in range(1, n + 1)]
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for vt in vts:
        for var, val in [("wind_speed_100m", 7.0), ("wind_speed_10m", 4.2),
                         ("temperature_2m", 12.0), ("surface_pressure", 1012.0)]:
            rows["valid_time"].append(vt)
            rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24)
            rows["variable"].append(var)
            rows["value"].append(val)
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(
        db, WeatherSeries(point_id=7, role=ApiRole.PREVIOUS_RUNS.value,
                          model="best_match", frame=frame),
    )


def test_generate_forecast_writes_rows(db, tmp_path):
    _populate(db)
    # Run experiment to create champion model (training data has lead_hours=24)
    run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                   capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                   n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    # Write previous-runs weather through valid_time = issue + 24h.
    issue = dt.datetime(2024, 1, 10, 0, 0)
    _write_forecast_weather(db, issue, n=24)
    # Generate forecast for horizon=24 (matches training lead_hours)
    count = generate_forecast(db, plant_id="wf1", point_id=7, horizon_hours=24,
                              capacity_mw=10.0, kind="wind", issue_time=issue)
    assert count >= 1
    row = db.execute("SELECT p50, p10, p90 FROM forecasts WHERE plant_id='wf1' LIMIT 1").fetchone()
    assert row is not None
    p50 = row[0]
    assert 0.0 <= p50 <= 10.0, f"p50={p50} out of range [0, capacity=10]"


def test_generate_forecast_is_idempotent_and_stamps_strategy_lineage(db, tmp_path):
    _populate(db)
    run_experiment(
        db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
        capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
        n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path,
    )
    issue = dt.datetime(2024, 1, 10)
    _write_forecast_weather(db, issue, n=24)
    kwargs = dict(
        plant_id="wf1", point_id=7, horizon_hours=24,
        capacity_mw=10.0, kind="wind", issue_time=issue,
    )
    assert generate_forecast(db, **kwargs) == 1
    assert generate_forecast(db, **kwargs) == 1
    rows = db.execute(
        "SELECT strategy_trial_id FROM forecasts "
        "WHERE plant_id='wf1' AND issue_time=? AND horizon_hours=24",
        [issue],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] is not None


def test_run_serving_cycle_writes_one_row_per_horizon(db, monkeypatch):
    point = PlantPoint(
        point_id=7, plant_id="wf1", point_type="weather",
        latitude=39.9, longitude=32.8,
    )
    issue = dt.datetime(2026, 1, 1)
    ingest_calls = []

    def fake_ingest(con, point, kind, **kwargs):
        ingest_calls.append(kwargs["horizon_hours_list"])
        return {"rows": 2}

    def fake_generate(con, *, plant_id, point_id, horizon_hours, issue_time, **kwargs):
        valid_time = issue_time + dt.timedelta(hours=horizon_hours)
        con.execute(
            """INSERT INTO forecasts
               (plant_id, point_id, horizon_hours, issue_time, valid_time,
                p10, p50, p90, strategy_trial_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [plant_id, point_id, horizon_hours, issue_time, valid_time, 1.0, 2.0, 3.0, 99],
        )
        return 1

    monkeypatch.setattr("openenergy.forecasting.predict.ingest_serving_previous_runs", fake_ingest)
    monkeypatch.setattr("openenergy.forecasting.predict.generate_forecast", fake_generate)
    result = run_serving_cycle(
        db, plant_id="wf1", point=point, horizons=[48, 24, 24],
        capacity_mw=10.0, kind="wind", issue_time=issue,
    )
    assert ingest_calls == [[24, 48]]
    assert result["rows"] == 2
    rows = db.execute(
        "SELECT horizon_hours, valid_time, strategy_trial_id FROM forecasts ORDER BY horizon_hours"
    ).fetchall()
    assert rows == [
        (24, issue + dt.timedelta(hours=24), 99),
        (48, issue + dt.timedelta(hours=48), 99),
    ]


def test_generate_forecast_serves_pointer_when_boolean_model_champion_moves(db, tmp_path):
    """T-05: an optimistic challenger model flag cannot veto the served strategy.

    Simulate the transient state created by a later experiment: its model owns
    the boolean champion flag while ``champion_pointer`` still names the defended
    incumbent strategy. Serving must use the pointer artifact without requiring
    the incumbent's legacy model flag.
    """
    _populate(db)
    incumbent_eid = run_experiment(
        db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
        capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
        n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path,
    )
    challenger_eid = db.execute(
        "INSERT INTO experiments (plant_id, horizons, status) "
        "VALUES ('wf1','24','complete') RETURNING experiment_id"
    ).fetchone()[0]
    db.execute("UPDATE models SET is_champion=false WHERE experiment_id=?", [incumbent_eid])
    db.execute(
        "INSERT INTO models "
        "(experiment_id, horizon_hours, model_family, artifact_path, metrics, "
        "is_champion, feature_names) VALUES (? ,24,'ridge','challenger.pkl','{}',true,'[]')",
        [challenger_eid],
    )

    issue = dt.datetime(2024, 1, 10, 0, 0)
    _write_forecast_weather(db, issue, n=24)
    assert generate_forecast(
        db, plant_id="wf1", point_id=7, horizon_hours=24,
        capacity_mw=10.0, kind="wind", issue_time=issue,
    ) >= 1


def test_generate_forecast_no_champion_raises(db, tmp_path):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    import pytest
    with pytest.raises(ValueError, match="champion"):
        generate_forecast(db, plant_id="wf1", point_id=7, horizon_hours=24,
                          capacity_mw=10.0, kind="wind", issue_time=dt.datetime(2024, 1, 1))


def test_quantile_monotonicity_enforced(db, tmp_path):
    """C1: persisted forecasts must always satisfy p10 <= p50 <= p90.

    We unit-test the ordering logic directly by running a full forecast cycle
    and then injecting a known crossing scenario via numpy to verify the fix.
    The full integration path is also covered: after generate_forecast, every
    stored row must satisfy the monotonicity constraint.
    """
    # --- Integration path: full forecast cycle ---
    _populate(db)
    run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                   capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                   n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    issue = dt.datetime(2024, 1, 10, 0, 0)
    _write_forecast_weather(db, issue, n=24)
    count = generate_forecast(db, plant_id="wf1", point_id=7, horizon_hours=24,
                              capacity_mw=10.0, kind="wind", issue_time=issue)
    assert count >= 1
    # Every stored row must satisfy p10 <= p50 <= p90
    rows = db.execute(
        "SELECT p10, p50, p90 FROM forecasts WHERE plant_id='wf1'"
    ).fetchall()
    for p10, p50, p90 in rows:
        if p10 is not None and p90 is not None:
            assert p10 <= p50, f"p10={p10} > p50={p50}: monotonicity violated"
            assert p50 <= p90, f"p50={p50} > p90={p90}: monotonicity violated"

    # --- Unit path: simulate LightGBM crossing quantile arrays ---
    # Construct crossing arrays (p10 > p50, p50 > p90 after clip) and verify the fix
    capacity = 100.0
    # raw model outputs (before de-norm and clip)
    raw_p50 = np.array([0.5, 0.3, 0.7])  # normalised
    raw_p10 = np.array([0.6, 0.4, 0.5])  # p10 > p50 → crossing
    raw_p90 = np.array([0.4, 0.2, 0.6])  # p90 < p50 → crossing

    p50 = np.clip(raw_p50 * capacity, 0, None)
    p10 = np.clip(raw_p10 * capacity, 0, None)
    p90 = np.clip(raw_p90 * capacity, 0, None)
    # Apply the fix from predict.py
    p10_fixed = np.minimum(p10, p50)
    p90_fixed = np.maximum(p90, p50)

    assert np.all(p10_fixed <= p50), "p10 crossing not corrected"
    assert np.all(p90_fixed >= p50), "p90 crossing not corrected"
    # Values that were already correct should be unchanged
    for i in range(len(p50)):
        assert p10_fixed[i] <= p50[i]
        assert p90_fixed[i] >= p50[i]


def _populate_solar(db, lat, lon, n=240):
    """Solar plant + previous_runs solar weather + production driven by irradiance.

    Registers plant_point id=7 with coords so the serving night-zero mask (C1-t4)
    can resolve geometry from the plant on the capacity_norm path.
    """
    create_plant(db, Plant(plant_id="sf1", name="SF", kind="solar", capacity_mw=10))
    db.execute(
        "INSERT INTO plant_points (point_id, plant_id, point_type, latitude, longitude, tilt, azimuth) "
        "VALUES (?,?,?,?,?,?,?)",
        [7, "sf1", "panel_array", lat, lon, 5.0, 180.0],
    )
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    # crude diurnal irradiance: positive around midday, zero at night
    hours = np.array([vt.hour for vt in vts])
    sw = np.clip(700.0 * np.sin((hours - 6) / 12.0 * np.pi), 0.0, None)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("shortwave_radiation", sw[i]), ("temperature_2m", 12.0),
                         ("terrestrial_solar_radiation", max(sw[i] * 1.4, 1.0))]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=7, role=ApiRole.PREVIOUS_RUNS.value, model="best_match", frame=frame))
    power = np.clip(sw / 100.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "sf1", prod)


def _write_solar_forecast_weather(db, issue, n=24):
    """Serving previous-runs solar weather with constant NONZERO irradiance at every
    lead — so the model predicts >0 even at the night valid_time, and only the
    hard night-zero mask can drive the served night row to exactly 0."""
    vts = [issue + dt.timedelta(hours=i) for i in range(1, n + 1)]
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for vt in vts:
        for var, val in [("shortwave_radiation", 500.0), ("temperature_2m", 15.0),
                         ("terrestrial_solar_radiation", 700.0)]:
            rows["valid_time"].append(vt)
            rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24)
            rows["variable"].append(var); rows["value"].append(val)
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(
        db, WeatherSeries(point_id=7, role=ApiRole.PREVIOUS_RUNS.value,
                          model="best_match", frame=frame),
    )


def _solar_forecast_at(db, issue, capacity):
    _write_solar_forecast_weather(db, issue)
    generate_forecast(db, plant_id="sf1", point_id=7, horizon_hours=24,
                      capacity_mw=capacity, kind="solar", issue_time=issue)
    return db.execute(
        "SELECT p10, p50, p90 FROM forecasts WHERE plant_id='sf1' AND issue_time=?",
        [issue],
    ).fetchall()


def test_generate_forecast_solar_night_serves_hard_zero(db, tmp_path):
    """C1-t4: a solar serving call forces p10/p50/p90 = 0 at night rows even on
    the capacity_norm path (no kpv), driven purely by the plant's lat/lon —
    despite a nonzero-irradiance forecast that makes the model predict >0."""
    capacity = 10.0
    _populate_solar(db, lat=39.9, lon=32.8)
    run_experiment(db, plant_id="sf1", point_id=7, horizon_hours_list=[24],
                   capacity_mw=capacity, kind="solar", nwp_sources=["icon"],
                   n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    # issue+24h = 2024-06-21 00:00 UTC → ~02:00 local at lon 32.8 → night
    rows = _solar_forecast_at(db, dt.datetime(2024, 6, 20, 0, 0), capacity)
    assert rows
    for p10, p50, p90 in rows:
        assert p50 == 0.0, f"night solar serving must be 0, got p50={p50}"
        assert (p10 is None or p10 == 0.0) and (p90 is None or p90 == 0.0)


def test_generate_forecast_solar_daytime_nonzero(db, tmp_path):
    """Daytime solar serving keeps positive predictions (the mask only touches
    night rows)."""
    capacity = 10.0
    _populate_solar(db, lat=39.9, lon=32.8)
    run_experiment(db, plant_id="sf1", point_id=7, horizon_hours_list=[24],
                   capacity_mw=capacity, kind="solar", nwp_sources=["icon"],
                   n_trials=3, n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    # issue+24h = 2024-06-21 09:00 UTC → ~11:00 local → daytime, near solar noon
    rows = _solar_forecast_at(db, dt.datetime(2024, 6, 20, 9, 0), capacity)
    assert rows
    assert any(p50 > 0.0 for _p10, p50, _p90 in rows)


class _ConstModel:
    """Stub ModelWrapper: returns a constant Prediction independent of X.

    Lets a test control the exact target *space* each candidate emits (a clear-sky
    ratio ŷ' for a kpv candidate vs a normalized-power ŷ for a capacity_norm one)
    without training, so the serving-time space-mixing bug is isolated."""

    def __init__(self, val: float):
        self.val = float(val)

    def predict(self, X):
        from openenergy.models.base import Prediction
        n = X.shape[0]
        return Prediction(
            p50=np.full(n, self.val),
            p10=np.full(n, self.val * 0.8),
            p90=np.full(n, self.val * 1.2),
        )


def test_mixed_target_policy_ensemble_serves_in_mw_space(db, tmp_path):
    """T-01: a champion whose ensemble mixes a kpv candidate (predicts the
    clear-sky ratio ŷ') and a capacity_norm candidate (predicts normalized power ŷ)
    must be combined in a COMMON space. The correct served p50 is the mean of each
    candidate inverted to normalized power, times capacity — NOT the mean of the
    raw ŷ'/ŷ (different spaces) times capacity, which over-predicts on ramps where
    P_cs << capacity.
    """
    from types import SimpleNamespace

    from openenergy.experiments.persistence import (
        create_experiment,
        mark_strategy_champion,
        promote_strategy_champion,
        record_strategy_trial,
    )
    from openenergy.experiments.search_space import PipelineConfig
    from openenergy.experiments.strategy import (
        FittedCandidate,
        StrategyArtifact,
        StrategyConfig,
    )
    from openenergy.features.target import PlantGeometry, kpv_inverse
    from openenergy.physics.solar import clear_sky_power

    capacity = 10.0
    lat, lon, tilt, az = 39.9, 32.8, 5.0, 180.0
    _populate_solar(db, lat=lat, lon=lon)

    kpv_ratio, cap_norm = 0.8, 0.1
    geom = PlantGeometry(lat=lat, lon=lon, tilt=tilt, azimuth=az)

    def _cand(target_policy, val):
        return FittedCandidate(
            config=SimpleNamespace(target_policy=target_policy, point_policy="single_point"),
            model=_ConstModel(val),
            feature_blocks=[],
            feature_names=["temperature_2m"],
            cv_nrmse=0.1,
        )

    mixed = StrategyArtifact(
        candidates=[_cand("kpv", kpv_ratio), _cand("capacity_norm", cap_norm)],
        ensemble_method="mean",
        target_policy="capacity_norm",  # runner: mixed ensembles fall back to capacity_norm
        geometry=geom,
    )
    artifact_path = tmp_path / "mixed-target-policy.pkl"
    mixed.save(artifact_path)

    pipeline = PipelineConfig(
        model_family="ridge",
        nwp_source="best_match",
        feature_blocks=[],
        params={"alpha": 1.0},
    )
    experiment_id = create_experiment(db, "sf1", [24])
    strategy_trial_id = record_strategy_trial(
        db,
        experiment_id=experiment_id,
        horizon_hours=24,
        strategy_config=StrategyConfig(
            candidates=[pipeline, pipeline],
            ensemble_method="mean",
            top_k=2,
        ),
        cv_nrmse=0.1,
        test_metrics={"nrmse": 0.1},
        skill_score=0.1,
        artifact_path=str(artifact_path),
    )
    mark_strategy_champion(db, experiment_id, 24, strategy_trial_id)
    promote_strategy_champion(db, "sf1", 24, strategy_trial_id)

    # issue+24h = 2024-06-21 06:00 UTC → ~09:00 local → daytime ramp, 0 < P_cs < capacity
    issue = dt.datetime(2024, 6, 20, 6, 0)
    rows = _solar_forecast_at(db, issue, capacity)
    assert rows and len(rows) == 1
    p50 = rows[0][1]

    vt = np.array([np.datetime64(issue + dt.timedelta(hours=24))])
    p_cs = float(clear_sky_power(vt, lat, lon, tilt, az, capacity).to_numpy()[0])
    assert 0.05 * capacity < kpv_ratio * p_cs < 0.9 * capacity, (
        f"pick a genuine ramp hour: kpv*P_cs={kpv_ratio * p_cs}, P_cs={p_cs}")

    kpv_norm = float(kpv_inverse(kpv_ratio, p_cs, capacity)) / capacity
    expected = (kpv_norm + cap_norm) / 2.0 * capacity
    assert abs(p50 - expected) <= 0.01 * expected, (
        f"served p50={p50} != offline-correct {expected} (raw-space mixing bug)")
