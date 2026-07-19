import datetime as dt

import numpy as np
import polars as pl

from openenergy.assets.repository import Plant, create_plant
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.features import FeatureConfig
from openenergy.experiments.assembly import assemble_pooled, assemble_raw, featurize
from openenergy.experiments.runner import evaluate_pooled, run_pooled_competitor


def _populate_multilead(db, plant_id="wfp", point_id=7, leads=(24, 48), n=240):
    """Seed a wind plant whose previous-runs weather exists at MULTIPLE leads for
    the same valid_times, so the frame can be pooled across leads. Production is a
    single series keyed by valid_time (shared by every lead)."""
    create_plant(db, Plant(plant_id=plant_id, name="WFP", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(0)
    ws = 5 + 3 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, n)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for lead in leads:
        # each lead carries a slightly degraded version of the truth (longer lead,
        # noisier), so bands should naturally widen at the longer lead.
        deg = 1.0 + 0.02 * (lead / 24.0)
        ws_lead = ws * deg + rng.normal(0, 0.1 * (lead / 24.0), n)
        for i, vt in enumerate(vts):
            for var, val in [("wind_speed_100m", ws_lead[i]), ("wind_speed_10m", ws_lead[i] * 0.6),
                             ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
                rows["valid_time"].append(vt)
                rows["issue_time"].append(vt - dt.timedelta(hours=lead))
                rows["lead_hours"].append(lead)
                rows["variable"].append(var)
                rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                                           model="best_match", frame=frame))
    power = np.clip((ws ** 3) / 1000.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)
    return vts


def test_assemble_pooled_adds_lead_time_hours_feature(db):
    _populate_multilead(db, leads=(24, 48))
    cfg = FeatureConfig(blocks=["wind_power"])
    pooled = assemble_pooled(db, plant_id="wfp", point_id=7, horizon_hours_list=[24, 48],
                             capacity_mw=10.0, kind="wind", feature_config=cfg)
    # lead_time_hours is a genuine feature column
    assert "lead_time_hours" in pooled.feature_names
    # single-lead featurization for reference (columns, minus the pooled lead column)
    raw = assemble_raw(db, plant_id="wfp", point_id=7, horizon_hours=24)
    single = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0)
    assert pooled.X.shape[1] == single.X.shape[1] + 1  # exactly one extra column
    # rows pooled across both leads
    assert pooled.X.shape[0] == single.X.shape[0] * 2
    # the lead_time_hours feature column matches the lead_time_hours bookkeeping array
    lead_col = pooled.X[:, pooled.feature_names.index("lead_time_hours")]
    np.testing.assert_array_equal(lead_col, pooled.lead_time_hours)
    assert set(np.unique(pooled.lead_time_hours).tolist()) == {24.0, 48.0}
    # X / y / valid_time / lead_time_hours all row-aligned
    assert (pooled.X.shape[0] == pooled.y.shape[0]
            == pooled.valid_time.shape[0] == pooled.lead_time_hours.shape[0])


def test_evaluate_pooled_no_cross_lead_leakage_and_per_lead_metrics(db):
    _populate_multilead(db, leads=(24, 48))
    cfg = FeatureConfig(blocks=["wind_power", "air_density"])
    pooled = assemble_pooled(db, plant_id="wfp", point_id=7, horizon_hours_list=[24, 48],
                             capacity_mw=10.0, kind="wind", feature_config=cfg)
    result = evaluate_pooled(pooled, params={"n_estimators": 60, "num_leaves": 15})
    # one metrics dict per lead
    assert set(result.per_lead.keys()) == {24.0, 48.0}
    for lead, tm in result.per_lead.items():
        assert np.isfinite(tm["nrmse"]) and tm["nrmse"] >= 0.0
        # per-lead-bucket calibration yields real bands → coverage/width recorded
        assert tm["coverage"] is not None
        assert tm["interval_width"] is not None and tm["interval_width"] > 0.0
    # NO cross-lead leakage: no valid_time appears in both train and test split.
    tr_times = set(pooled.valid_time[result.train_idx].tolist())
    te_times = set(pooled.valid_time[result.test_idx].tolist())
    assert tr_times.isdisjoint(te_times)


def test_run_pooled_competitor_records_leaderboard_verdict(db):
    from openenergy.experiments.scenario_memory import (
        pooled_strategy_family,
        record_scenario_run,
        refresh_scenario_leaderboard,
        scenario_leaderboard,
        ScenarioRun,
    )

    _populate_multilead(db, plant_id="wfp", point_id=7, leads=(24, 48))
    # A pre-existing per-horizon champion at 24h to compete against.
    record_scenario_run(db, ScenarioRun(
        experiment_id=999, plant_id="wfp", asset_kind="wind", horizon_hours=24,
        strategy_family="wind_topk_none", nrmse=0.30, skill_score=0.2, is_champion=True,
    ))
    refresh_scenario_leaderboard(db)

    result = run_pooled_competitor(db, plant_id="wfp", point_id=7,
                                   horizon_hours_list=[24, 48], capacity_mw=10.0,
                                   kind="wind", feature_blocks=["wind_power", "air_density"],
                                   params={"n_estimators": 60, "num_leaves": 15})
    fam = pooled_strategy_family("wind")
    # pooled competitor recorded a scenario_run per lead under its own family
    rows = db.execute(
        "SELECT horizon_hours, nrmse, strategy_family FROM scenario_runs WHERE strategy_family=?",
        [fam],
    ).fetchall()
    assert {r[0] for r in rows} == {24, 48}
    assert all(r[1] is not None for r in rows)
    # the leaderboard now shows BOTH the per-horizon family AND the pooled family
    board24 = scenario_leaderboard(db, asset_kind="wind", horizon_hours=24)
    families = {b["strategy_family"] for b in board24}
    assert fam in families and "wind_topk_none" in families
    # the returned result exposes the per-lead pooled-vs-per-horizon verdict
    assert set(result.per_lead.keys()) == {24.0, 48.0}
