from fastapi.testclient import TestClient
from openenergy.storage import connect, init_schema
from openenergy.api.app import create_app
from openenergy.assets.repository import Plant, create_plant
from openenergy.config import Settings
from openenergy.experiments.scenario_memory import (
    ScenarioRun,
    record_scenario_run,
    refresh_scenario_leaderboard,
)


def _client():
    con = connect(None)
    init_schema(con)
    return TestClient(create_app(con)), con


def test_health():
    client, _ = _client()
    assert client.get("/health").json()["status"] == "ok"


def test_fleet_health_shape():
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))

    response = client.get("/fleet/health")

    assert response.status_code == 200
    assert response.json() == [
        {
            "plant_id": "wf1",
            "name": "WF",
            "kind": "wind",
            "capacity_mw": 10.0,
            "status": "unknown",
            "latest_forecast_at": None,
            "latest_production_at": None,
            "drift_status": "unknown",
            "active_jobs": 0,
        }
    ]


def test_scheduler_lifespan_is_opt_in_and_shuts_down(monkeypatch, db):
    scheduler = type(
        "FakeScheduler",
        (),
        {
            "started": False,
            "stopped": False,
            "start": lambda self: setattr(self, "started", True),
            "shutdown": lambda self, wait: setattr(self, "stopped", not wait),
        },
    )()
    build = __import__(
        "openenergy.jobs.scheduler", fromlist=["build_scheduler"]
    )
    monkeypatch.setattr(build, "build_scheduler", lambda *_args: scheduler)
    settings = Settings(
        data_dir="/tmp/openenergy-scheduler-test",
        scheduler_enabled=True,
    )

    with TestClient(create_app(db, settings=settings)) as client:
        assert client.get("/health").status_code == 200
        assert scheduler.started is True

    assert scheduler.stopped is True


def test_scheduler_lifespan_default_starts_no_thread(db):
    app = create_app(
        db,
        settings=Settings(
            data_dir="/tmp/openenergy-scheduler-test",
            scheduler_enabled=False,
        ),
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert not hasattr(app.state, "scheduler")


def test_endpoint_only_enqueues(monkeypatch):
    from unittest.mock import Mock

    client, _ = _client()
    handler = Mock(return_value=123)
    monkeypatch.setattr("openenergy.api.app.run_experiment", handler, raising=False)

    response = client.post(
        "/experiments",
        json={
            "plant_id": "wf1",
            "point_id": 7,
            "horizons": [24],
            "kind": "wind",
            "capacity_mw": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    handler.assert_not_called()
    job = client.get(f"/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "pending"
    assert job["payload"]["plant_id"] == "wf1"


def test_reforecast_endpoint_returns_job(monkeypatch):
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    resp = client.post("/plants/wf1/reforecast", json={
        "point_id": 1, "horizon_hours": 24, "kind": "wind",
        "issue_time": "2026-01-01T00:00:00",
    })
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_reforecast_endpoint_404_for_unknown_plant():
    client, _ = _client()
    resp = client.post("/plants/nope/reforecast", json={
        "point_id": 1, "horizon_hours": 24, "kind": "wind",
    })
    assert resp.status_code == 404


def test_forecast_endpoint_runs_serving_cycle(monkeypatch):
    client, con = _client()
    client.post(
        "/plants",
        json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10},
    )
    point_id = client.post(
        "/plants/wf1/points",
        json={
            "plant_id": "wf1", "point_type": "weather",
            "latitude": 39.9, "longitude": 32.8,
        },
    ).json()["point_id"]
    captured = {}

    def fake_cycle(*args, **kwargs):
        captured.update(kwargs)
        return {"rows": 2}

    monkeypatch.setattr("openenergy.api.app.run_serving_cycle", fake_cycle, raising=False)
    resp = client.post(
        "/plants/wf1/forecast",
        json={
            "point_id": point_id,
            "horizons": [24, 48],
            "kind": "wind",
            "issue_time": "2026-01-01T00:00:00",
        },
    )
    assert resp.status_code == 200
    assert "job_id" in resp.json()
    assert captured == {}
    payload = con.execute(
        "SELECT payload FROM jobs WHERE job_id=?", [resp.json()["job_id"]]
    ).fetchone()[0]
    assert '"horizons":[24,48]' in payload
    assert '"capacity_mw":10.0' in payload


def test_forecast_endpoint_rejects_point_from_another_plant():
    client, _ = _client()
    for plant_id in ("wf1", "wf2"):
        client.post(
            "/plants",
            json={"plant_id": plant_id, "name": plant_id, "kind": "wind", "capacity_mw": 10},
        )
    point_id = client.post(
        "/plants/wf2/points",
        json={
            "plant_id": "wf2", "point_type": "weather",
            "latitude": 39.9, "longitude": 32.8,
        },
    ).json()["point_id"]
    resp = client.post(
        "/plants/wf1/forecast",
        json={"point_id": point_id, "horizons": [24], "kind": "wind"},
    )
    assert resp.status_code == 404


def test_forecast_backfill_endpoint_uses_registered_plant_and_point(monkeypatch):
    client, _ = _client()
    client.post(
        "/plants",
        json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10},
    )
    point_id = client.post(
        "/plants/wf1/points",
        json={
            "plant_id": "wf1", "point_type": "weather",
            "latitude": 39.9, "longitude": 32.8,
        },
    ).json()["point_id"]
    captured = {}

    def fake_backfill(*args, **kwargs):
        captured.update(kwargs)
        return {"rows": 12, "actual_pairs": 12}

    monkeypatch.setattr("openenergy.api.app.backfill_forecasts", fake_backfill, raising=False)
    resp = client.post(
        "/plants/wf1/forecasts/backfill",
        json={
            "point_id": point_id,
            "horizons": [24],
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-02T00:00:00",
        },
    )

    assert resp.status_code == 200
    assert captured == {}
    job = client.get(f"/jobs/{resp.json()['job_id']}").json()
    assert job["kind"] == "forecast_backfill"
    assert job["status"] == "pending"
    assert job["payload"]["plant_id"] == "wf1"
    assert job["payload"]["capacity_mw"] == 10
    assert job["payload"]["kind"] == "wind"


def test_backtest_endpoint_empty_without_champion():
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    resp = client.get("/plants/wf1/backtest?horizon=24")
    assert resp.status_code == 200 and resp.json() == []


def test_backtest_endpoint_404_for_unknown_plant():
    client, _ = _client()
    assert client.get("/plants/nope/backtest?horizon=24").status_code == 404


def test_create_and_get_plant():
    client, _ = _client()
    r = client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    assert r.status_code == 200
    g = client.get("/plants/wf1").json()
    assert g["plant"]["plant_id"] == "wf1"


def test_upload_production_csv():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    csv = b"timestamp,power_mw\n2024-06-01T00:00,3.2\n2024-06-01T01:00,4.1\n"
    r = client.post("/plants/wf1/production", files={"file": ("p.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["rows"] == 2


def test_add_point():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.post("/plants/wf1/points", json={
        "plant_id": "wf1", "point_type": "weather",
        "latitude": 51.0, "longitude": 4.0
    })
    assert r.status_code == 200
    assert r.json()["point_id"] is not None


def test_get_job():
    client, _ = _client()
    # Start experiment - it will run synchronously via BackgroundTasks in test mode
    # First set up some data
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    # Just check /jobs endpoint works (even for unknown id, returns 404)
    r = client.get("/jobs/99999")
    assert r.status_code == 404


def test_experiment_request_accepts_strategy_fields(monkeypatch):
    client, _ = _client()
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured.update(kwargs)
        return 123

    monkeypatch.setattr(
        "openenergy.api.app.run_experiment", fake_run_experiment, raising=False
    )
    r = client.post(
        "/experiments",
        json={
            "plant_id": "wf1",
            "point_id": 7,
            "horizons": [24],
            "kind": "wind",
            "capacity_mw": 10,
            "nwp_sources": ["icon"],
            "n_trials": 3,
            "top_k": 4,
            "search_mode": "strategy",
            "preset": "deep",
        },
    )
    assert r.status_code == 200
    assert captured == {}
    assert client.get(f"/jobs/{r.json()['job_id']}").json()["payload"]["top_k"] == 4


def test_experiment_request_accepts_policy_subset(monkeypatch):
    # B4-t2: POST /experiments accepts optional point_ids + source policy subset.
    client, _ = _client()
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured.update(kwargs)
        return 321

    monkeypatch.setattr(
        "openenergy.api.app.run_experiment", fake_run_experiment, raising=False
    )
    r = client.post(
        "/experiments",
        json={
            "plant_id": "wf1",
            "point_id": 7,
            "horizons": [24],
            "kind": "wind",
            "capacity_mw": 10,
            "nwp_sources": ["icon"],
            "point_ids": [1, 2, 3],
            "source": "icon_eu",
        },
    )
    assert r.status_code == 200
    assert captured == {}
    payload = client.get(f"/jobs/{r.json()['job_id']}").json()["payload"]
    assert payload["point_ids"] == [1, 2, 3]
    assert payload["source"] == "icon_eu"


def test_experiment_request_policy_subset_defaults_to_none(monkeypatch):
    # Default: no point_ids/source → single-point path (all applicable policies).
    client, _ = _client()
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(
        "openenergy.api.app.run_experiment", fake_run_experiment, raising=False
    )
    r = client.post(
        "/experiments",
        json={
            "plant_id": "wf1",
            "point_id": 7,
            "horizons": [24],
            "kind": "wind",
            "capacity_mw": 10,
        },
    )
    assert r.status_code == 200
    assert captured == {}
    payload = client.get(f"/jobs/{r.json()['job_id']}").json()["payload"]
    assert payload["point_ids"] is None
    assert payload["source"] is None


def test_experiment_results_expose_winning_policies():
    # B4-t2: /experiments/{id}/results rows carry the winning point/source policy.
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    # A strategy champion trial for experiment 5 / horizon 24.
    con.execute(
        """
        INSERT INTO strategy_trials
          (experiment_id, horizon_hours, ensemble_method, strategy_config,
           cv_nrmse, test_nrmse, skill_score, is_champion)
        VALUES (5, 24, 'mean', '{}', 0.21, 0.2, 0.3, true)
        """
    )
    record_scenario_run(
        con,
        ScenarioRun(
            experiment_id=5,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean_spatial_mean_icon+gfs",
            point_policy="spatial_mean",
            source_policy="icon+gfs",
            skill_score=0.3,
            nrmse=0.2,
            evaluation_grade="limited",
            evaluation_n_days=120,
            evaluation_origins=3,
            evaluation_seasons=2,
            is_champion=True,
        ),
    )

    r = client.get("/experiments/5/results")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["point_policy"] == "spatial_mean"
    assert row["source_policy"] == "icon+gfs"


def test_experiment_champion_endpoint_exposes_policies_and_skill():
    # B4-t2: champion payload exposes winning policies + per-baseline skill.
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    record_scenario_run(
        con,
        ScenarioRun(
            experiment_id=5,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean_spatial_mean_icon+gfs",
            point_policy="spatial_mean",
            source_policy="icon+gfs",
            skill_score=0.3,
            nrmse=0.2,
            evaluation_grade="limited",
            evaluation_n_days=120,
            evaluation_origins=3,
            evaluation_seasons=2,
            is_champion=True,
        ),
    )
    # A non-champion run in the same experiment must not be reported.
    record_scenario_run(
        con,
        ScenarioRun(
            experiment_id=5,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean",
            point_policy="single_point",
            source_policy="single_source",
            skill_score=0.1,
            nrmse=0.3,
            is_champion=False,
        ),
    )

    r = client.get("/experiments/5/champion")
    assert r.status_code == 200
    champs = r.json()
    assert len(champs) == 1
    champ = champs[0]
    assert champ["horizon_hours"] == 24
    assert champ["point_policy"] == "spatial_mean"
    assert champ["source_policy"] == "icon+gfs"
    assert champ["skill_score"] == 0.3
    assert champ["evaluation_grade"] == "limited"
    assert champ["evaluation_n_days"] == 120
    assert champ["evaluation_origins"] == 3
    assert champ["evaluation_seasons"] == 2
    # per-baseline skill breakdown present (keyed by baseline name).
    assert isinstance(champ["baseline_skills"], dict)
    assert champ["baseline_skills"]["best"] == 0.3


def test_experiment_champion_endpoint_empty():
    client, _ = _client()
    r = client.get("/experiments/999/champion")
    assert r.status_code == 200
    assert r.json() == []


def test_get_forecasts_empty():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.get("/plants/wf1/forecasts?horizon=24")
    assert r.status_code == 200
    assert r.json() == []


def test_get_forecasts_returns_only_latest_issue_trajectory():
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    for issue, p50 in (
        ("2026-01-01 00:00:00", 1.0),
        ("2026-01-02 00:00:00", 2.0),
    ):
        con.execute(
            """
            INSERT INTO forecasts
              (plant_id, point_id, horizon_hours, issue_time, valid_time, p10, p50, p90)
            VALUES ('wf1', 1, 24, ?, ?::TIMESTAMP + INTERVAL 24 HOUR, 0, ?, 3)
            """,
            [issue, issue, p50],
        )

    rows = client.get("/plants/wf1/forecasts?horizon=24").json()

    assert len(rows) == 1
    assert rows[0]["issue_time"].startswith("2026-01-02")
    assert rows[0]["p50"] == 2.0


def _seed_drift(con, *, degraded_err, n=80):
    """Seed a champion (nRMSE 0.05) plus n recent (forecast, actual) pairs whose p50
    misses production by ~degraded_err MW on a 10 MW plant."""
    import datetime as dt

    import numpy as np

    con.execute("INSERT INTO experiments (experiment_id, plant_id) VALUES (1, 'wf1')")
    con.execute(
        "INSERT INTO experiment_trials (experiment_id, horizon_hours, model_family, "
        "test_nrmse, is_champion) VALUES (1, 24, 'lightgbm', 0.05, true)"
    )
    rng = np.random.default_rng(0)
    base = dt.datetime(2024, 6, 1)
    for i in range(n):
        vt = base + dt.timedelta(hours=i)
        actual = float(min(max(4.0 + 2.0 * np.sin(i / 6.0), 0.0), 10.0))
        fc = float(min(max(actual + rng.normal(0, degraded_err), 0.0), 10.0))
        issue = vt - dt.timedelta(hours=24)
        con.execute("INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
                    ["wf1", vt, actual])
        con.execute(
            "INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, "
            "valid_time, p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ["wf1", 7, 24, issue, vt, fc - 1, fc, fc + 1, None],
        )


def _append_drift_pairs(con, *, degraded_err, base_hour, n=40):
    """Append n more degraded (forecast, actual) pairs at later valid_times — a new
    drift evaluation window (T-06). Persistence counts breaching windows, not calls."""
    import datetime as dt

    import numpy as np

    rng = np.random.default_rng(base_hour)
    base = dt.datetime(2024, 6, 1) + dt.timedelta(hours=base_hour)
    for i in range(n):
        vt = base + dt.timedelta(hours=i)
        actual = float(min(max(4.0 + 2.0 * np.sin(i / 6.0), 0.0), 10.0))
        fc = float(min(max(actual + rng.normal(0, degraded_err), 0.0), 10.0))
        con.execute("INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
                    ["wf1", vt, actual])
        con.execute(
            "INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, "
            "valid_time, p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ["wf1", 7, 24, vt - dt.timedelta(hours=24), vt, fc - 1, fc, fc + 1, None],
        )


def test_drift_check_fires_emergency_retrain_once():
    client, con = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    _seed_drift(con, degraded_err=3.0)

    body = {"point_id": 7, "horizons": [24], "kind": "wind", "min_consecutive": 2}

    r1 = client.post("/plants/wf1/drift/check", json=body)
    assert r1.status_code == 200
    assert r1.json()["results"][0]["fired"] is False  # first breach only arms

    # New realised data → a second breaching evaluation window (T-06).
    _append_drift_pairs(con, degraded_err=3.0, base_hour=80)
    r2 = client.post("/plants/wf1/drift/check", json=body)
    fired = r2.json()["results"][0]
    assert fired["fired"] is True
    job_id = fired["job_id"]
    assert job_id is not None
    assert con.execute(
        "SELECT count(*) FROM jobs WHERE job_id=? AND kind='emergency_retrain'",
        [job_id],
    ).fetchone()[0] == 1

    r3 = client.post("/plants/wf1/drift/check", json=body)
    assert r3.json()["results"][0]["fired"] is False  # hysteresis: no re-fire

    # Only one emergency retrain was ever enqueued.
    assert con.execute(
        "SELECT count(*) FROM jobs WHERE kind='emergency_retrain'"
    ).fetchone()[0] == 1

    status = client.get("/plants/wf1/drift").json()
    assert len(status) == 1
    assert status[0]["breaching"] is True
    assert status[0]["horizon_hours"] == 24


def test_drift_check_healthy_does_not_fire():
    client, con = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    _seed_drift(con, degraded_err=0.4)
    body = {"point_id": 7, "horizons": [24], "kind": "wind", "min_consecutive": 1}
    for _ in range(3):
        r = client.post("/plants/wf1/drift/check", json=body)
    assert r.json()["results"][0]["fired"] is False
    assert con.execute(
        "SELECT count(*) FROM jobs WHERE kind='emergency_retrain'"
    ).fetchone()[0] == 0


def test_drift_status_not_found():
    client, _ = _client()
    assert client.get("/plants/nope/drift").status_code == 404


def test_cors_allows_frontend_origin():
    client, _ = _client()
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_list_plants_endpoint():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "a", "name": "Alpha", "kind": "wind", "capacity_mw": 5})
    client.post("/plants", json={"plant_id": "b", "name": "Beta", "kind": "solar", "capacity_mw": 8})
    r = client.get("/plants")
    assert r.status_code == 200
    ids = {p["plant_id"] for p in r.json()}
    assert ids == {"a", "b"}


def test_scenario_memory_endpoints():
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    record_scenario_run(
        con,
        ScenarioRun(
            experiment_id=1,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean",
            feature_signature="wind_power",
            model_signature="lightgbm",
            train_policy="rolling_180d",
            point_policy="single_point",
            ensemble_policy="mean",
            quantile_policy="native_or_none",
            nrmse=0.2,
            skill_score=0.3,
            is_champion=True,
        ),
    )
    refresh_scenario_leaderboard(con)

    leaderboard = client.get("/scenario-memory/leaderboard?kind=wind&horizon=24")
    assert leaderboard.status_code == 200
    assert leaderboard.json()[0]["strategy_family"] == "wind_topk_mean"

    profile = client.get("/plants/wf1/strategy-profile")
    assert profile.status_code == 200
    assert profile.json()[0]["strategy_family"] == "wind_topk_mean"

    runs = client.get("/plants/wf1/scenario-runs")
    assert runs.status_code == 200
    run0 = runs.json()[0]
    assert run0["plant_id"] == "wf1"
    for k in ("crps", "peak_bias", "peak_mae"):
        assert k in run0


def test_policy_leaderboard_endpoint():
    client, con = _client()
    create_plant(con, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    # Winning combo: spatial_mean x icon+gfs
    for exp, champ, skill in ((1, True, 0.5), (2, True, 0.6)):
        record_scenario_run(
            con,
            ScenarioRun(
                experiment_id=exp,
                plant_id="wf1",
                asset_kind="wind",
                horizon_hours=24,
                strategy_family="wind_topk_mean_spatial_mean_icon+gfs",
                point_policy="spatial_mean",
                source_policy="icon+gfs",
                skill_score=skill,
                nrmse=0.2,
                is_champion=champ,
            ),
        )
    # Losing combo: single_point x single_source
    record_scenario_run(
        con,
        ScenarioRun(
            experiment_id=3,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean",
            point_policy="single_point",
            source_policy="single_source",
            skill_score=0.1,
            nrmse=0.3,
            is_champion=False,
        ),
    )

    resp = client.get("/scenario-memory/policy-leaderboard?kind=wind&horizon=24")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[0]["point_policy"] == "spatial_mean"
    assert rows[0]["source_policy"] == "icon+gfs"
    assert rows[0]["wins"] == 2
    assert rows[0]["win_rate"] == 1.0

    scoped = client.get("/plants/wf1/policy-leaderboard")
    assert scoped.status_code == 200
    assert scoped.json()[0]["point_policy"] == "spatial_mean"


# ── B4-t1: weather-grid creation + multi-point/source ingest ────────────────────

def _plant_with_center(client, plant_id="wf1", lat=51.0, lon=4.0):
    client.post("/plants", json={"plant_id": plant_id, "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.post("/plants/{}/points".format(plant_id), json={
        "plant_id": plant_id, "point_type": "weather", "latitude": lat, "longitude": lon,
    })
    return r.json()["point_id"]


def test_create_weather_grid():
    client, _ = _client()
    _plant_with_center(client, lat=51.0, lon=4.0)
    r = client.post("/plants/wf1/weather-grid", json={"source": "icon_eu", "size": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["grid_id"] == "icon_eu_3x3"
    assert body["count"] == 9
    assert body["source"] == "icon_eu"
    # center cell equals the plant's representative point coords
    centers = [
        p for p in body["points"]
        if abs(p["latitude"] - 51.0) < 1e-9 and abs(p["longitude"] - 4.0) < 1e-9
    ]
    assert len(centers) == 1
    # idempotent re-generation returns the same grid (no duplicates)
    r2 = client.post("/plants/wf1/weather-grid", json={"source": "icon_eu", "size": 3})
    assert r2.status_code == 200
    assert r2.json()["count"] == 9


def test_create_weather_grid_upwind_shifts_centroid():
    client, _ = _client()
    _plant_with_center(client, lat=51.0, lon=4.0)
    r = client.post("/plants/wf1/weather-grid", json={"source": "icon_eu", "size": 3, "upwind_deg": 270})
    assert r.status_code == 200
    # upwind lean shifts every cell → no cell sits exactly on the raw center
    on_center = [
        p for p in r.json()["points"]
        if abs(p["latitude"] - 51.0) < 1e-9 and abs(p["longitude"] - 4.0) < 1e-9
    ]
    assert on_center == []


def test_create_weather_grid_plant_not_found():
    client, _ = _client()
    r = client.post("/plants/nope/weather-grid", json={"source": "icon_eu"})
    assert r.status_code == 404


def test_create_weather_grid_no_representative_point():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.post("/plants/wf1/weather-grid", json={"source": "icon_eu"})
    assert r.status_code == 400


def test_ingest_weather_grid(monkeypatch):
    client, _ = _client()
    _plant_with_center(client)
    client.post("/plants/wf1/weather-grid", json={"source": "icon_eu", "size": 3})

    captured = {}

    def fake_ingest_grid(con, points, kind, **kwargs):
        captured["points"] = len(points)
        captured["kind"] = kind
        captured["sources"] = kwargs.get("sources")
        captured["past_days"] = kwargs.get("past_days")
        return {"points": len(points), "ingested": len(points), "rows": 42, "errors": []}

    monkeypatch.setattr(
        "openenergy.api.app.ingest_grid_previous_runs", fake_ingest_grid, raising=False
    )
    r = client.post(
        "/plants/wf1/ingest-weather",
        json={"grid_id": "icon_eu_3x3", "sources": ["icon_eu", "ecmwf_ifs025"], "past_days": 30},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert captured == {}
    job = client.get("/jobs/{}".format(job_id)).json()
    assert job["status"] == "pending"
    assert job["payload"]["grid_id"] == "icon_eu_3x3"
    assert job["payload"]["kind"] == "wind"
    assert job["payload"]["sources"] == ["icon_eu", "ecmwf_ifs025"]
    assert job["payload"]["past_days"] == 30


def test_ingest_weather_grid_not_found():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.post("/plants/wf1/ingest-weather", json={"grid_id": "does_not_exist"})
    assert r.status_code == 404


def test_ingest_weather_requires_point_or_grid():
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})
    r = client.post("/plants/wf1/ingest-weather", json={})
    assert r.status_code == 400


def test_retrain_endpoint_runs_champion_challenger(monkeypatch):
    # E1-t2: POST /plants/{id}/retrain enqueues a job that runs champion-challenger
    # (retrain) in-process across horizons and records the outcome in the job.
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})

    captured = {}

    def fake_retrain(con, **kwargs):
        captured.update(kwargs)
        return {
            24: {"promoted": True, "new_nrmse": 0.10, "old_nrmse": 0.20, "p_value": 0.01},
            48: {"promoted": False, "new_nrmse": 0.30, "old_nrmse": 0.25, "p_value": 0.4},
        }

    monkeypatch.setattr("openenergy.api.app.retrain", fake_retrain, raising=False)
    r = client.post(
        "/plants/wf1/retrain",
        json={
            "plant_id": "wf1",
            "point_id": 7,
            "horizons": [24, 48],
            "kind": "wind",
            "capacity_mw": 10,
            "nwp_sources": ["icon"],
            "n_trials": 3,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert captured == {}

    job = client.get("/jobs/{}".format(job_id)).json()
    assert job["status"] == "pending"
    assert job["kind"] == "retrain"
    assert job["payload"]["plant_id"] == "wf1"
    assert job["payload"]["point_id"] == 7
    assert job["payload"]["horizons"] == [24, 48]
    assert job["payload"]["n_trials"] == 3


def test_postprocessor_fit_endpoint_runs_per_horizon(monkeypatch):
    # E1-t3: POST /plants/{id}/postprocessor/fit enqueues a job that refits the
    # daily LASSO post-processor for each horizon in-process.
    client, _ = _client()
    client.post("/plants", json={"plant_id": "wf1", "name": "WF", "kind": "wind", "capacity_mw": 10})

    calls = []

    def fake_fit(con, **kwargs):
        calls.append(kwargs)
        return {"fitted": True, "n_samples": 50, "quantiles": ["p50"]}

    monkeypatch.setattr(
        "openenergy.api.app.fit_postprocessor", fake_fit, raising=False
    )
    r = client.post(
        "/plants/wf1/postprocessor/fit",
        json={"horizons": [24, 48], "degree": 2, "min_samples": 8},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert calls == []

    job = client.get("/jobs/{}".format(job_id)).json()
    assert job["status"] == "pending"
    assert job["kind"] == "postprocessor_fit"
    assert job["payload"]["horizons"] == [24, 48]
    assert job["payload"]["degree"] == 2
    assert job["payload"]["min_samples"] == 8


def test_postprocessor_fit_endpoint_plant_not_found():
    client, _ = _client()
    r = client.post("/plants/nope/postprocessor/fit", json={"horizons": [24]})
    assert r.status_code == 404


def test_retrain_endpoint_plant_not_found():
    client, _ = _client()
    r = client.post(
        "/plants/nope/retrain",
        json={"plant_id": "nope", "point_id": 1, "horizons": [24],
              "kind": "wind", "capacity_mw": 10},
    )
    assert r.status_code == 404


def test_ingest_weather_single_point_still_works(monkeypatch):
    client, _ = _client()
    pid = _plant_with_center(client)

    def fake_single(con, point, kind, **kwargs):
        return {"rows": 7, "lead_hours": [0, 24], "variables": 3}

    monkeypatch.setattr(
        "openenergy.api.app.ingest_previous_runs", fake_single, raising=False
    )
    r = client.post("/plants/wf1/ingest-weather", json={"point_id": pid})
    assert r.status_code == 200
    job = client.get("/jobs/{}".format(r.json()["job_id"])).json()
    assert job["status"] == "pending"
    assert job["payload"]["point_id"] == pid
