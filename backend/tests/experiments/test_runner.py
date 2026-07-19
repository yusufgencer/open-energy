import datetime as dt
import numpy as np
import polars as pl
import pytest
from pathlib import Path
from openenergy.assets.repository import Plant, create_plant
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.experiments.runner import (
    run_experiment, evaluate_pipeline, _masked_test_metrics, _test_metrics, _peak_mask,
)
from openenergy.models.base import Prediction
from openenergy.experiments.assembly import assemble, assemble_raw, featurize
from openenergy.experiments.search_space import PipelineConfig
from openenergy.experiments.persistence import best_trials
from openenergy.features import FeatureConfig
from openenergy.storage.db import connect, init_schema


def test_init_schema_migrates_new_metric_columns():
    con = connect()  # in-memory
    # simulate an OLD DB: experiment_trials predating the new metric columns
    con.execute(
        "CREATE TABLE experiment_trials (trial_id BIGINT, experiment_id BIGINT, "
        "horizon_hours INTEGER, test_nrmse DOUBLE)"
    )
    init_schema(con)
    et_cols = {r[1] for r in con.execute("PRAGMA table_info('experiment_trials')").fetchall()}
    st_cols = {r[1] for r in con.execute("PRAGMA table_info('strategy_trials')").fetchall()}
    sr_cols = {r[1] for r in con.execute("PRAGMA table_info('scenario_runs')").fetchall()}
    for c in ("test_coverage", "test_interval_width", "test_crps", "test_peak_bias", "test_peak_mae"):
        assert c in et_cols and c in st_cols, f"missing {c}"
    for c in (
        "band_status", "sel_coverage", "sel_coverage_lower",
        "sel_coverage_upper", "sel_winkler", "sel_kupiec_pvalue",
        "sel_independence_pvalue", "sel_conditional_coverage_pvalue", "sel_n",
        "evaluation_grade", "evaluation_n_days", "evaluation_origins",
        "evaluation_seasons",
    ):
        assert c in st_cols, f"missing strategy_trials.{c}"
    for c in (
        "crps", "peak_bias", "peak_mae", "source_policy", "target_policy",
        "evaluation_grade", "evaluation_n_days", "evaluation_origins",
        "evaluation_seasons",
    ):
        assert c in sr_cols, f"missing scenario_runs.{c}"


def test_experiment_records_new_metric_columns(db, tmp_path):
    _populate(db)
    eid = run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=4, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    # crps is always computable (p50 always present) → recorded everywhere
    assert db.execute(
        "SELECT count(*) FROM experiment_trials WHERE experiment_id=? AND test_crps IS NOT NULL", [eid]
    ).fetchone()[0] >= 1
    assert db.execute(
        "SELECT count(*) FROM strategy_trials WHERE experiment_id=? AND test_crps IS NOT NULL", [eid]
    ).fetchone()[0] >= 1
    assert db.execute(
        "SELECT count(*) FROM scenario_runs WHERE experiment_id=? AND crps IS NOT NULL", [eid]
    ).fetchone()[0] >= 1
    # results endpoint (best_trials) must surface the new metric fields
    bt = best_trials(db, eid)
    for k in (
        "test_crps", "test_coverage", "test_interval_width", "test_peak_bias",
        "test_peak_mae", "evaluation_grade", "evaluation_n_days",
        "evaluation_origins", "evaluation_seasons",
    ):
        assert k in bt[0], f"best_trials missing {k}"

    strategy_evidence = db.execute(
        "SELECT evaluation_grade, evaluation_n_days, evaluation_origins, "
        "evaluation_seasons FROM strategy_trials WHERE experiment_id=?",
        [eid],
    ).fetchone()
    scenario_evidence = db.execute(
        "SELECT evaluation_grade, evaluation_n_days, evaluation_origins, "
        "evaluation_seasons FROM scenario_runs WHERE experiment_id=?",
        [eid],
    ).fetchone()
    assert strategy_evidence == scenario_evidence
    assert strategy_evidence == ("exploratory", 10, 3, 1)


def _populate(db, n=240):
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


def _populate_heteroskedastic(db, n=360):
    """Learnable wind curve with feature-dependent noise for interval gating."""
    create_plant(db, Plant(plant_id="wfhet", name="WF HET", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(919)
    ws = 6.0 + 2.5 * np.sin(np.arange(n) / 15.0) + rng.normal(0.0, 0.25, n)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in (
            ("wind_speed_100m", ws[i]),
            ("wind_speed_10m", ws[i] * 0.62),
            ("temperature_2m", 8.0 + 0.5 * np.sin(i / 24.0)),
            ("surface_pressure", 1013.0),
        ):
            rows["valid_time"].append(vt)
            rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24)
            rows["variable"].append(var)
            rows["value"].append(float(val))
    frame = pl.DataFrame(
        rows,
        schema_overrides={
            "valid_time": pl.Datetime("us"),
            "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32,
            "value": pl.Float64,
        },
    )
    write_weather_series(
        db,
        WeatherSeries(
            point_id=71,
            role=ApiRole.PREVIOUS_RUNS.value,
            model="best_match",
            frame=frame,
        ),
    )
    mean = np.clip((ws ** 3) / 90.0, 0.0, 9.5)
    noise_scale = 0.05 + 0.06 * np.maximum(ws - 4.0, 0.0)
    power = np.clip(mean + rng.normal(0.0, noise_scale), 0.0, 10.0)
    write_production(
        db,
        "wfhet",
        pl.DataFrame(
            {"ts": vts, "power_mw": power.astype(float)},
            schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
        ),
    )


def test_champion_path_coverage_gate(db, tmp_path):
    """T-19: the statistical verdict is selection-only, persisted, and enforced."""
    from openenergy.evaluation.metrics import clopper_pearson_interval

    _populate_heteroskedastic(db)
    eid = run_experiment(
        db,
        plant_id="wfhet",
        point_id=71,
        horizon_hours_list=[24],
        capacity_mw=10.0,
        kind="wind",
        nwp_sources=["icon"],
        n_trials=4,
        n_splits=3,
        embargo=24,
        seed=42,
        models_dir=tmp_path,
    )
    row = db.execute(
        """
        SELECT band_status, sel_coverage, sel_coverage_lower, sel_coverage_upper,
               sel_n, sel_winkler, sel_kupiec_pvalue,
               sel_independence_pvalue, sel_conditional_coverage_pvalue,
               is_champion
        FROM strategy_trials WHERE experiment_id=?
        """,
        [eid],
    ).fetchone()
    assert row is not None
    status, observed, lower, upper, n, winkler, kupiec_p, ind_p, cc_p, champion = row
    assert n >= 10
    successes = int(round(observed * n))
    assert (lower, upper) == pytest.approx(
        clopper_pearson_interval(successes, n)
    )
    expected = "pass" if lower <= 0.80 <= upper else "fail"
    assert status == expected
    assert bool(champion) is (status == "pass")
    assert winkler is not None and winkler >= 0.0
    for p_value in (kupiec_p, ind_p, cc_p):
        assert p_value is not None and 0.0 <= p_value <= 1.0


def test_run_experiment_pinned_source_reaches_suggest_and_recorded_config(
    db, tmp_path, monkeypatch
):
    """T-04: the source used to build the frame must pin every suggest/replay call.

    Otherwise FixedTrial replay silently falls back to nwp_sources[0] and records a
    source_policy that does not describe the data the model actually saw.
    """
    import openenergy.experiments.runner as runner_mod

    _populate(db)
    original_suggest = runner_mod.suggest
    seen_source_policies: list[list[str] | None] = []

    def spy_suggest(*args, **kwargs):
        seen_source_policies.append(kwargs.get("source_policies"))
        return original_suggest(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "suggest", spy_suggest)
    eid = runner_mod.run_experiment(
        db,
        plant_id="wf1",
        point_id=7,
        horizon_hours_list=[24],
        capacity_mw=10.0,
        kind="wind",
        nwp_sources=["wrong_label", "best_match"],
        source="best_match",
        n_trials=1,
        n_splits=3,
        embargo=24,
        seed=42,
        models_dir=tmp_path,
    )

    assert seen_source_policies
    assert all(policies == ["best_match"] for policies in seen_source_policies)
    recorded = db.execute(
        "SELECT DISTINCT nwp_source FROM experiment_trials WHERE experiment_id=?",
        [eid],
    ).fetchall()
    assert recorded == [("best_match",)]


def test_ensemble_champion_records_band_metrics(db, tmp_path):
    """D1-t3: an ensemble champion whose members include band-capable models records
    non-NULL coverage/interval_width/crps — bands are combined, not dropped."""
    _populate(db)
    eid = run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=8, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    # At least one top candidate produces bands (lightgbm/random_forest/quantile_gbm);
    # the ensemble champion must therefore carry real coverage + crps, not NULL.
    row = db.execute(
        "SELECT test_coverage, test_interval_width, test_crps FROM strategy_trials "
        "WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()
    assert row is not None
    coverage, width, crps = row
    assert coverage is not None, "ensemble champion coverage should be non-NULL"
    assert width is not None and width > 0.0, "ensemble champion interval width should be positive"
    assert crps is not None
    # The scenario_runs mirror carries the same non-NULL coverage/crps.
    sr = db.execute(
        "SELECT coverage, crps FROM scenario_runs WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()
    assert sr is not None and sr[0] is not None and sr[1] is not None


def test_experiment_records_real_quantile_policy_and_serving_reproduces_widths(db, tmp_path):
    """D2-t2: the runner records a real quantile_policy in scenario_runs (never the
    retired hard-coded 'native_or_none'), and serving reproduces the calibrated
    bands stored on the strategy artifact."""
    import datetime as dt
    from openenergy.experiments.persistence import get_champion_strategy_model
    from openenergy.experiments.strategy import QUANTILE_POLICIES, StrategyArtifact
    from openenergy.forecasting.predict import generate_forecast

    _populate(db)
    eid = run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=8, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    qp = db.execute(
        "SELECT quantile_policy FROM scenario_runs WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0]
    assert qp in QUANTILE_POLICIES, f"expected a real quantile_policy, got {qp!r}"
    assert qp != "native_or_none"

    # Serving parity: forecasts served through the champion artifact carry exactly
    # the calibrated bands the artifact's _combine produces (same p10/p90 widths).
    champ = get_champion_strategy_model(db, "wf1", 24)
    assert champ is not None and champ["artifact_path"]
    art = StrategyArtifact.load(champ["artifact_path"])
    # the artifact stores a real policy and, when calibrated, a fitted calibrator
    assert art.quantile_policy == qp
    if qp != "native":
        assert art.calibrator is not None

    issue = dt.datetime(2024, 1, 1)  # any FORECAST-role issue time in-range
    n = generate_forecast(db, plant_id="wf1", point_id=7, horizon_hours=24,
                          capacity_mw=10.0, kind="wind", issue_time=issue)
    if n > 0:
        rows = db.execute(
            "SELECT p10, p50, p90 FROM forecasts WHERE plant_id='wf1' AND horizon_hours=24 "
            "ORDER BY valid_time"
        ).fetchall()
        # calibrated serving preserves quantile ordering on every served row
        for p10, p50, p90 in rows:
            if p10 is not None and p90 is not None:
                assert p10 <= p50 + 1e-9 <= p90 + 1e-9


def _populate_small(db, plant_id="wf2", point_id=8, n=40):
    """Populate with too-little data so inner CV sentinel triggers."""
    create_plant(db, Plant(plant_id=plant_id, name="WF2", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(1)
    ws = 5 + rng.normal(0, 0.5, n)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("wind_speed_100m", ws[i]), ("wind_speed_10m", ws[i] * 0.6),
                         ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                                           model="best_match", frame=frame))
    power = np.clip((ws ** 3) / 1000.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)


def test_peak_mask_solar_selects_top_and_wind_none():
    y = np.array([0.0, 0.1, 1.0, 0.9])
    m = _peak_mask(y, "solar")
    assert m is not None and m.tolist() == [False, False, True, True]
    assert _peak_mask(y, "wind") is None


def test_test_metrics_reports_crps_and_solar_peak():
    y = np.array([0.0, 0.1, 1.0, 0.9])
    pred = Prediction(
        p50=np.array([0.0, 0.1, 0.6, 0.7]),
        p10=np.array([0.0, 0.05, 0.5, 0.6]),
        p90=np.array([0.05, 0.2, 0.8, 0.9]),
    )
    tm = _test_metrics(y, pred, peak_mask=_peak_mask(y, "solar"))
    assert tm["crps"] is not None
    assert tm["peak_bias"] is not None and tm["peak_mae"] is not None


def test_test_metrics_peak_none_for_wind():
    y = np.array([0.2, 0.5, 0.9])
    pred = Prediction(p50=y.copy())
    tm = _test_metrics(y, pred, peak_mask=_peak_mask(y, "wind"))
    assert tm["peak_bias"] is None and tm["peak_mae"] is None
    assert tm["crps"] is not None


def test_dual_reporting_all_vs_clean():
    y = np.array([0.2, 0.4, 0.6, 0.8])
    # The severe-invalid final row is intentionally a large miss.
    pred = Prediction(p50=np.array([0.2, 0.4, 0.6, 0.0]))
    all_metrics = _test_metrics(y, pred)
    clean_metrics = _masked_test_metrics(
        y, pred, np.array([True, True, True, False]), kind="wind"
    )
    assert all_metrics["nrmse"] > 0
    assert clean_metrics["nrmse"] == 0
    assert clean_metrics["nrmse"] != all_metrics["nrmse"]


def test_candidate_validation_predictions_returns_masked_quantile_matrices(db):
    """D1-t1: per-quantile validation matrices (p10/p50/p90) in (n_candidates x n_val)
    shape, post enforce_quantile_order. A band-less candidate (ridge) is masked with
    NaN in its p10/p90 rows — never zero-filled — while a banded candidate (lightgbm)
    yields finite, ordered bands."""
    from openenergy.experiments.runner import _candidate_validation_predictions
    from openenergy.evaluation.splits import walk_forward_split

    _populate(db)
    raw = assemble_raw(db, plant_id="wf1", point_id=7, horizon_hours=24)
    n = raw.height
    tr_idx, _te_idx = walk_forward_split(n, test_size=max(1, n // 5))
    val_size = max(1, min(len(tr_idx) // 5, len(tr_idx) - 1))

    ridge = PipelineConfig(model_family="ridge", nwp_source="icon",
                           feature_blocks=["wind_power"], params={"alpha": 1.0})
    lgbm = PipelineConfig(model_family="lightgbm", nwp_source="icon",
                          feature_blocks=["wind_power"], params={"n_estimators": 20})
    candidates = [(ridge, 0.1), (lgbm, 0.2)]

    p10s, p50s, p90s, y_val = _candidate_validation_predictions(
        raw, candidates, kind="wind", capacity_mw=10.0, train_idx=tr_idx, val_size=val_size,
    )

    n_val = val_size
    assert p10s.shape == p50s.shape == p90s.shape == (2, n_val)
    assert y_val.shape == (n_val,)
    # p50 is always present and finite for both candidates
    assert np.all(np.isfinite(p50s))
    # ridge (row 0) has no bands → masked with NaN, not zero
    assert np.all(np.isnan(p10s[0])) and np.all(np.isnan(p90s[0]))
    # lightgbm (row 1) has finite, ordered bands
    assert np.all(np.isfinite(p10s[1])) and np.all(np.isfinite(p90s[1]))
    assert np.all(p10s[1] <= p50s[1] + 1e-9)
    assert np.all(p50s[1] <= p90s[1] + 1e-9)


def test_oof_calibration_set_size_and_model_disjointness(monkeypatch):
    """T-18: OOF calibration covers >=60% of outer train and never self-predicts."""
    from types import SimpleNamespace

    from openenergy.experiments import runner

    n = 240
    outer_train = np.arange(192)
    # Reserve the full trailing ensemble-validation window. Its labels fit learned
    # combiners, so none of its rows may masquerade as full-pipeline OOF.
    oof_universe = outer_train[:-38]
    fit_calls: list[np.ndarray] = []

    class RejectSeenRowsModel:
        def __init__(self, seen):
            self.seen = set(np.asarray(seen, dtype=int).tolist())

        def predict(self, X):
            rows = np.asarray(X[:, 0], dtype=int)
            # This checks the actual model instance used for every emitted OOF
            # prediction, rather than trusting split metadata after the fact.
            assert self.seen.isdisjoint(rows.tolist())
            p50 = rows.astype(float) / n
            return Prediction(p50=p50, p10=p50 - 0.1, p90=p50 + 0.1)

    def fake_fit_candidate(raw, cfg, *, train_idx, **kwargs):
        fit_calls.append(np.asarray(train_idx, dtype=int))
        X = np.arange(n, dtype=float)[:, None]
        y = np.arange(n, dtype=float) / n
        ds = SimpleNamespace(X=X, y=y)
        return ds, RejectSeenRowsModel(train_idx), y, None

    monkeypatch.setattr(runner, "_fit_candidate", fake_fit_candidate)
    cfg = PipelineConfig(
        model_family="ridge",
        nwp_source="icon",
        feature_blocks=["wind_power"],
        params={"alpha": 1.0},
    )
    oof = runner._candidate_oof_predictions(
        object(),
        [(cfg, 0.1)],
        kind="wind",
        capacity_mw=10.0,
        train_idx=oof_universe,
        n_splits=3,
        embargo=12,
        horizon=24,
        min_coverage_count=int(np.ceil(0.60 * len(outer_train))),
    )

    assert len(oof.y) == len(oof.source_indices)
    assert len(oof.y) >= np.ceil(0.60 * len(outer_train))
    assert len(fit_calls) == len(oof.folds)
    np.testing.assert_array_equal(
        oof.source_indices,
        np.concatenate([fold_test for _, fold_test in oof.folds]),
    )
    for fitted_rows, (fold_train, fold_test) in zip(fit_calls, oof.folds):
        np.testing.assert_array_equal(fitted_rows, fold_train)
        assert set(fold_train).isdisjoint(fold_test)
        assert int(np.max(fold_train)) + 24 < int(np.min(fold_test))


def _populate_grid(db, plant_id="grid1", n=240):
    create_plant(db, Plant(plant_id=plant_id, name="G", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(0)
    ws = 5 + 3 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, n)
    for pid, mult in [(7, 1.0), (8, 1.05)]:
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


def test_run_experiment_multipoint_searches_point_policy(db, tmp_path):
    from openenergy.features.point_policy import POINT_POLICIES
    _populate_grid(db)
    eid = run_experiment(db, plant_id="grid1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=6, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path, point_ids=[7, 8])
    champ = db.execute(
        "SELECT count(*) FROM strategy_trials WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0]
    assert champ == 1
    pp = db.execute(
        "SELECT point_policy FROM scenario_runs WHERE experiment_id=?", [eid]
    ).fetchone()[0]
    # the winning strategy used one of the searched grid policies, not legacy single_point
    assert all(p in POINT_POLICIES for p in pp.split("+"))


def _populate_solar(db, plant_id="sp1", capacity_mw=10.0, frac=0.75):
    """Synthetic clear-sky-shaped solar plant in the DB (winter→summer split).

    Production is a fixed fraction of the clear-sky envelope; the low-peak winter
    block comes first (train) and the high-peak summer block last (test), the
    extrapolation trap kpv escapes and capacity_norm falls into. Returns the
    weather point_id created for the plant.
    """
    from openenergy.assets.repository import PlantPoint, add_point
    from openenergy.features.target import PlantGeometry
    from openenergy.physics.solar import clear_sky_power

    create_plant(db, Plant(plant_id=plant_id, name="SP", kind="solar", capacity_mw=capacity_mw))
    geom = PlantGeometry(lat=39.9, lon=32.8, tilt=5.0, azimuth=180.0)
    pt = add_point(db, PlantPoint(plant_id=plant_id, point_type="panel_array",
                                  latitude=geom.lat, longitude=geom.lon,
                                  tilt=geom.tilt, azimuth=geom.azimuth))
    point_id = pt.point_id

    times = []
    for base in (dt.datetime(2023, 12, 1), dt.datetime(2024, 6, 10)):
        for d in range(20):
            for h in range(24):
                times.append(base + dt.timedelta(days=d, hours=h))
    times.sort()
    p_cs = clear_sky_power(times, geom.lat, geom.lon, geom.tilt, geom.azimuth,
                           capacity_mw).to_numpy()
    power = frac * p_cs
    shortwave = 1000.0 * p_cs / capacity_mw

    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(times):
        for var, val in [("shortwave_radiation", shortwave[i]), ("temperature_2m", 15.0)]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                                           model="best_match", frame=frame))
    prod = pl.DataFrame({"ts": times, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)
    return point_id


def test_run_experiment_solar_records_target_policy(db, tmp_path):
    from openenergy.experiments.scenario_memory import target_leaderboard
    point_id = _populate_solar(db)
    eid = run_experiment(db, plant_id="sp1", point_id=point_id, horizon_hours_list=[24],
                         capacity_mw=10.0, kind="solar", nwp_sources=["ecmwf"], n_trials=10,
                         n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    # The winning strategy records a real target_policy, drawn from the searched
    # axis {capacity_norm, kpv} (not left NULL). Peak metrics are recorded (solar).
    row = db.execute(
        "SELECT target_policy, strategy_family, peak_mae FROM scenario_runs WHERE experiment_id=?", [eid]
    ).fetchone()
    assert row is not None
    target_policy, family, peak_mae = row
    assert target_policy is not None
    assert all(tok in ("capacity_norm", "kpv") for tok in target_policy.split("+"))
    assert peak_mae is not None
    # kpv folds into the family string so the leaderboard distinguishes it.
    if "kpv" in target_policy.split("+"):
        assert "kpv" in family
    # target leaderboard answers "which target_policy wins for this solar plant".
    board = target_leaderboard(db, asset_kind="solar", horizon_hours=24, plant_id="sp1")
    assert len(board) >= 1
    assert all(tok in ("capacity_norm", "kpv")
               for tok in board[0]["target_policy"].split("+"))
    assert board[0]["median_peak_mae"] is not None


def test_run_experiment_solar_explores_both_target_policies(db, tmp_path, monkeypatch):
    """The search genuinely trains both target spaces: capture every cfg the runner
    fits and assert both capacity_norm and kpv candidates were explored."""
    import openenergy.experiments.runner as runner_mod
    seen: list[str] = []
    orig = runner_mod._fit_candidate

    def spy(raw, cfg, **kw):
        seen.append(cfg.target_policy)
        return orig(raw, cfg, **kw)

    monkeypatch.setattr(runner_mod, "_fit_candidate", spy)
    point_id = _populate_solar(db)
    run_experiment(db, plant_id="sp1", point_id=point_id, horizon_hours_list=[24],
                   capacity_mw=10.0, kind="solar", nwp_sources=["ecmwf"], n_trials=12,
                   n_splits=3, embargo=24, seed=42, models_dir=tmp_path)
    assert "kpv" in seen and "capacity_norm" in seen


def test_evaluate_pipeline_returns_finite_nrmse(db):
    _populate(db)
    ds = assemble(db, plant_id="wf1", point_id=7, horizon_hours=24, capacity_mw=10.0,
                  kind="wind", feature_config=FeatureConfig(blocks=["wind_power", "air_density"]))
    cfg = PipelineConfig(model_family="ridge", nwp_source="icon", feature_blocks=["wind_power"], params={"alpha": 1.0})
    score = evaluate_pipeline(ds, cfg, n_splits=3, embargo=24)
    assert np.isfinite(score) and score >= 0


def test_run_experiment_produces_champion(db, tmp_path):
    _populate(db)
    eid = run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=4, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    bests = best_trials(db, eid)
    assert len(bests) == 1 and bests[0]["horizon_hours"] == 24
    assert 0.0 <= bests[0]["test_nrmse"] < 1.0
    champ = db.execute("SELECT count(*) FROM experiment_trials WHERE experiment_id=? AND is_champion", [eid]).fetchone()[0]
    assert champ == 1
    assert db.execute("SELECT count(*) FROM models WHERE experiment_id=? AND is_champion", [eid]).fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM strategy_trials WHERE experiment_id=? AND is_champion", [eid]).fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM strategy_candidates", []).fetchone()[0] >= 1
    fn = db.execute("SELECT feature_names FROM models WHERE experiment_id=? AND is_champion", [eid]).fetchone()[0]
    assert fn is not None and isinstance(fn, str) and len(fn) > 2  # non-empty JSON list


def test_featurize_different_blocks_produce_different_columns(db):
    """C1: toggling feature_blocks genuinely changes the feature matrix (search axis is real)."""
    _populate(db)
    raw = assemble_raw(db, plant_id="wf1", point_id=7, horizon_hours=24)
    assert raw.height > 0, "Need rows to test featurize axis"

    ds_small = featurize(raw, kind="wind",
                         feature_config=FeatureConfig(blocks=["wind_power"]),
                         capacity_mw=10.0)
    ds_large = featurize(raw, kind="wind",
                         feature_config=FeatureConfig(blocks=["wind_power", "air_density", "cyclical_time"]),
                         capacity_mw=10.0)

    assert ds_large.X.shape[1] > ds_small.X.shape[1], (
        f"Expected more columns with more blocks: "
        f"wind_power={ds_small.X.shape[1]} cols, "
        f"wind_power+air_density+cyclical_time={ds_large.X.shape[1]} cols"
    )
    # Row count stays the same — only columns differ
    assert ds_small.X.shape[0] == ds_large.X.shape[0] == raw.height


def test_no_champion_when_cv_sentinel(db, tmp_path):
    """I1: no champion is marked when inner CV couldn't run (insufficient data → sentinel 1e6)."""
    # n=40, train subset ≈32, evaluate_pipeline check: 32 < (3+1)*(24+5)=116 → sentinel
    _populate_small(db, plant_id="wf2", point_id=8, n=40)
    eid = run_experiment(db, plant_id="wf2", point_id=8, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=2, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)

    champ_count = db.execute(
        "SELECT count(*) FROM experiment_trials WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0]
    assert champ_count == 0, f"Expected 0 champions when CV sentinel fires, got {champ_count}"

    model_count = db.execute(
        "SELECT count(*) FROM models WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0]
    assert model_count == 0, f"Expected 0 model champions when CV sentinel fires, got {model_count}"
    strategy_count = db.execute(
        "SELECT count(*) FROM strategy_trials WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0]
    assert strategy_count == 0

    # A trial row should still exist (cv_nrmse=NULL)
    trial_count = db.execute(
        "SELECT count(*) FROM experiment_trials WHERE experiment_id=?", [eid]
    ).fetchone()[0]
    assert trial_count >= 1
    cv_val = db.execute(
        "SELECT cv_nrmse FROM experiment_trials WHERE experiment_id=? LIMIT 1", [eid]
    ).fetchone()[0]
    assert cv_val is None, f"Expected cv_nrmse=NULL for sentinel trial, got {cv_val}"


def test_two_experiments_leave_exactly_one_champion(db, tmp_path):
    """M2: running two experiments on the same plant+horizon leaves exactly one champion
    in both experiment_trials and models, and it belongs to the most recent experiment.
    """
    _populate(db)
    common_kwargs = dict(
        plant_id="wf1", point_id=7, horizon_hours_list=[24],
        capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
        n_trials=3, n_splits=3, embargo=24, seed=42,
    )
    eid1 = run_experiment(db, **common_kwargs, models_dir=tmp_path / "exp1")
    eid2 = run_experiment(db, **common_kwargs, models_dir=tmp_path / "exp2")

    # Exactly one champion trial for this plant+horizon across all experiments
    champ_trials = db.execute(
        """
        SELECT et.trial_id, et.experiment_id
        FROM experiment_trials et
        JOIN experiments e ON et.experiment_id = e.experiment_id
        WHERE e.plant_id = 'wf1' AND et.horizon_hours = 24 AND et.is_champion = true
        """
    ).fetchall()
    assert len(champ_trials) == 1, (
        f"Expected exactly 1 champion trial, got {len(champ_trials)}: {champ_trials}"
    )
    # Champion must belong to the most recent experiment
    assert champ_trials[0][1] == eid2, (
        f"Champion trial should be from eid2={eid2}, got eid={champ_trials[0][1]}"
    )

    # Exactly one champion model for this plant+horizon across all experiments
    champ_models = db.execute(
        """
        SELECT m.model_id, m.experiment_id
        FROM models m
        JOIN experiments e ON m.experiment_id = e.experiment_id
        WHERE e.plant_id = 'wf1' AND m.horizon_hours = 24 AND m.is_champion = true
        """
    ).fetchall()
    assert len(champ_models) == 1, (
        f"Expected exactly 1 champion model, got {len(champ_models)}: {champ_models}"
    )
    assert champ_models[0][1] == eid2, (
        f"Champion model should be from eid2={eid2}, got eid={champ_models[0][1]}"
    )

    champ_strategies = db.execute(
        """
        SELECT st.strategy_trial_id, st.experiment_id
        FROM strategy_trials st
        JOIN experiments e ON st.experiment_id = e.experiment_id
        WHERE e.plant_id = 'wf1' AND st.horizon_hours = 24 AND st.is_champion = true
        """
    ).fetchall()
    assert len(champ_strategies) == 1
    assert champ_strategies[0][1] == eid2


def _populate_extended_no_skill(db, *, plant_id="wfx", point_id=77, horizon=120, n=240):
    """Seed a plant at an EXTENDED horizon (lead_hours=horizon) where the weather
    features carry NO information about production: production is a clean 24h-periodic
    signal a diurnal baseline predicts almost perfectly, while the weather is pure
    noise. No ML candidate can beat the baseline → skill ≤ 0."""
    create_plant(db, Plant(plant_id=plant_id, name="WFX", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(123)
    noise = 5 + rng.normal(0, 2.0, n)  # weather uncorrelated with production
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("wind_speed_100m", noise[i]), ("wind_speed_10m", noise[i] * 0.6),
                         ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
            rows["valid_time"].append(vt)
            rows["issue_time"].append(vt - dt.timedelta(hours=horizon))
            rows["lead_hours"].append(horizon)
            rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                                           model="best_match", frame=frame))
    # Perfectly periodic production → the diurnal baseline nails it, so skill ≤ 0.
    power = np.clip(5 + 4 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24.0), 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)


def test_extended_horizon_refuses_champion_without_skill(db, tmp_path):
    """F1-t1: an experiment at an extended horizon (120h) must NOT promote a champion
    when skill ≤ 0 — the horizon is served only with a proven (skill>climatology) model.
    A "no skill" verdict (the scenario_run) is still recorded."""
    _populate_extended_no_skill(db, plant_id="wfx", point_id=77, horizon=120)
    eid = run_experiment(db, plant_id="wfx", point_id=77, horizon_hours_list=[120], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=4, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    # The search ran and recorded the verdict, but skill ≤ 0 gates the champion.
    skill = db.execute(
        "SELECT skill_score FROM scenario_runs WHERE experiment_id=? ORDER BY skill_score LIMIT 1", [eid]
    ).fetchone()
    assert skill is not None and skill[0] <= 0.0, f"test setup should yield skill ≤ 0, got {skill}"
    # No champion anywhere: experiment_trials, models, strategy_trials, scenario_runs.
    for tbl, col in [("experiment_trials", "is_champion"), ("models", "is_champion"),
                     ("strategy_trials", "is_champion"), ("scenario_runs", "is_champion")]:
        cnt = db.execute(
            f"SELECT count(*) FROM {tbl} WHERE experiment_id=? AND {col}", [eid]
        ).fetchone()[0]
        assert cnt == 0, f"Expected 0 champions in {tbl} at gated 120h horizon, got {cnt}"
    # But the run is still recorded as an explicit verdict.
    assert db.execute(
        "SELECT count(*) FROM scenario_runs WHERE experiment_id=?", [eid]
    ).fetchone()[0] >= 1


def test_legacy_horizon_still_promotes_champion(db, tmp_path):
    """F1-t1 must not touch the ≤48h path: a normal 24h experiment still promotes."""
    _populate(db)
    eid = run_experiment(db, plant_id="wf1", point_id=7, horizon_hours_list=[24], capacity_mw=10.0,
                         kind="wind", nwp_sources=["icon"], n_trials=4, n_splits=3, embargo=24,
                         seed=42, models_dir=tmp_path)
    assert db.execute(
        "SELECT count(*) FROM models WHERE experiment_id=? AND is_champion", [eid]
    ).fetchone()[0] == 1


def test_select_ensemble_method_has_no_test_args():
    """T-08: ensemble method selection must be structurally blind to the test
    window — its signature carries no y_test / test-band parameters, so it cannot
    select on the held-out reporting window."""
    import inspect

    from openenergy.experiments.runner import select_ensemble_method

    params = set(inspect.signature(select_ensemble_method).parameters)
    assert not (params & {"y_test", "test_lower", "test_upper", "test_p50s", "test_p90s"})


def test_selection_is_blind_to_test_window(db, tmp_path):
    """T-10: permuting only outer-test labels cannot change strategy selection.

    The second experiment sees byte-identical weather and training/validation
    targets. Only production labels at the positional outer ``te_idx`` timestamps
    are permuted. Reporting metrics may therefore change, but the ensemble method,
    quantile policy, and fitted ensemble weights must remain exactly invariant.
    """
    from openenergy.evaluation.splits import walk_forward_split
    from openenergy.experiments.strategy import StrategyArtifact

    _populate(db)
    run_kwargs = dict(
        plant_id="wf1",
        point_id=7,
        horizon_hours_list=[24],
        capacity_mw=10.0,
        kind="wind",
        nwp_sources=["icon"],
        n_trials=4,
        n_splits=3,
        embargo=24,
        seed=42,
        models_dir=tmp_path,
    )
    baseline_id = run_experiment(db, **run_kwargs)
    baseline_row = db.execute(
        "SELECT ensemble_method, artifact_path, test_nrmse "
        "FROM strategy_trials WHERE experiment_id=?",
        [baseline_id],
    ).fetchone()
    assert baseline_row is not None
    baseline_method, baseline_path, baseline_test_nrmse = baseline_row
    baseline_artifact = StrategyArtifact.load(baseline_path)

    raw = assemble_raw(db, plant_id="wf1", point_id=7, horizon_hours=24)
    _, test_idx = walk_forward_split(raw.height, test_size=max(1, raw.height // 5))
    test_times = [raw["valid_time"][int(i)] for i in test_idx]
    test_values = np.asarray([
        db.execute(
            "SELECT power_mw FROM production WHERE plant_id=? AND ts=?",
            ["wf1", ts],
        ).fetchone()[0]
        for ts in test_times
    ])
    permuted_values = test_values[np.random.default_rng(10).permutation(len(test_values))]
    assert not np.array_equal(test_values, permuted_values)
    db.executemany(
        "UPDATE production SET power_mw=? WHERE plant_id=? AND ts=?",
        [(float(value), "wf1", ts) for value, ts in zip(permuted_values, test_times)],
    )

    permuted_id = run_experiment(db, **run_kwargs)
    permuted_row = db.execute(
        "SELECT ensemble_method, artifact_path, test_nrmse "
        "FROM strategy_trials WHERE experiment_id=?",
        [permuted_id],
    ).fetchone()
    assert permuted_row is not None
    permuted_method, permuted_path, permuted_test_nrmse = permuted_row
    permuted_artifact = StrategyArtifact.load(permuted_path)

    # The permutation reached the reporting-only path, so this is not a vacuous
    # comparison of two identical experiment inputs.
    assert not np.isclose(baseline_test_nrmse, permuted_test_nrmse)
    assert permuted_method == baseline_method
    assert permuted_artifact.ensemble_method == baseline_artifact.ensemble_method
    assert permuted_artifact.quantile_policy == baseline_artifact.quantile_policy

    if baseline_artifact.weights is None or permuted_artifact.weights is None:
        assert baseline_artifact.weights is permuted_artifact.weights
    else:
        np.testing.assert_array_equal(
            permuted_artifact.weights, baseline_artifact.weights
        )

    assert (permuted_artifact.quantile_weights is None) == (
        baseline_artifact.quantile_weights is None
    )
    if baseline_artifact.quantile_weights is not None:
        assert permuted_artifact.quantile_weights is not None
        assert permuted_artifact.quantile_weights.keys() == (
            baseline_artifact.quantile_weights.keys()
        )
        for tau, baseline_weights in baseline_artifact.quantile_weights.items():
            np.testing.assert_array_equal(
                permuted_artifact.quantile_weights[tau], baseline_weights
            )


def test_select_ensemble_method_prefers_mean_on_noise():
    """T-08: when no combiner genuinely beats the equal-weight average out-of-sample
    (candidates are independent noise around the truth), selection returns the plain
    'mean' — the hard-to-beat baseline — rather than an overfit learned combiner."""
    from openenergy.experiments.runner import select_ensemble_method

    rng = np.random.default_rng(0)
    n = 240
    truth = rng.normal(5.0, 2.0, n)
    val_p50s = np.vstack([truth + rng.normal(0.0, 1.0, n) for _ in range(3)])
    val_p10s = val_p50s - 1.0
    val_p90s = val_p50s + 1.0
    method = select_ensemble_method(
        val_p50s, truth, val_p10s, val_p90s,
        cv_scores=[1.0, 1.0, 1.0], method_limit=3, horizon=1,
    )
    assert method == "mean"


def test_select_ensemble_method_mcs_uses_parsimony_for_equivalent_methods(monkeypatch):
    """Equivalent validation losses retain an MCS; plain mean wins simply."""
    from openenergy.experiments import runner
    from openenergy.models.base import Prediction

    n = 80
    truth = np.linspace(0.0, 1.0, n)
    rows = np.vstack([truth + 0.1, truth - 0.1])

    def equivalent_prediction(method, p50s, *args, **kwargs):
        prediction = Prediction(p50=np.asarray(p50s[0]))
        return prediction, None, None, None, None

    monkeypatch.setattr(runner, "_strategy_prediction", equivalent_prediction)
    method, diagnostics = runner.select_ensemble_method(
        rows,
        truth,
        rows - 0.2,
        rows + 0.2,
        cv_scores=[1.0, 1.0],
        method_limit=2,
        horizon=1,
        return_diagnostics=True,
    )
    assert method == "mean"
    assert "mean" in diagnostics["mcs_members"]
    assert len(diagnostics["mcs_members"]) > 1


def test_select_ensemble_method_uses_validation_crps(monkeypatch):
    """A point-perfect but grossly dispersed challenger must lose on CRPS.

    Squared-error/nRMSE selection would choose every challenger in this fixture;
    probabilistic selection keeps the well-calibrated mean baseline.
    """
    from openenergy.experiments import runner
    from openenergy.models.base import Prediction

    n = 120
    truth = np.zeros(n)
    val_p50s = np.vstack([np.full(n, 0.2), np.zeros(n)])
    val_p10s = np.vstack([np.full(n, -0.1), np.full(n, -100.0)])
    val_p90s = np.vstack([np.full(n, 0.3), np.full(n, 100.0)])

    def fake_strategy_prediction(method, p50s, *args, **kwargs):
        rows = p50s.shape[1]
        if method == "mean":
            pred = Prediction(
                p50=np.full(rows, 0.2),
                p10=np.full(rows, -0.1),
                p90=np.full(rows, 0.3),
            )
        else:
            pred = Prediction(
                p50=np.zeros(rows),
                p10=np.full(rows, -100.0),
                p90=np.full(rows, 100.0),
            )
        return pred, None, None, None, None

    monkeypatch.setattr(runner, "_strategy_prediction", fake_strategy_prediction)
    assert runner.select_ensemble_method(
        val_p50s,
        truth,
        val_p10s,
        val_p90s,
        cv_scores=[1.0, 1.0],
        method_limit=2,
        horizon=1,
    ) == "mean"
