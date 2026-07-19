import datetime as dt
import numpy as np
import polars as pl
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.assets.repository import Plant, create_plant
from openenergy.features import FeatureConfig
from openenergy.experiments.assembly import featurize
from openenergy.datasets.horizon import build_multipoint_dataset
from openenergy.forecasting.assembly import forecast_matrix, forecast_matrix_multipoint


def _fc_series(point_id, issue, h, n=24):
    vts = [issue + dt.timedelta(hours=i) for i in range(1, n + 1)]
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for vt in vts:
        for var, val in [("wind_speed_100m", 7.0), ("temperature_2m", 12.0), ("surface_pressure", 1012.0)]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=h))
            rows["lead_hours"].append(h)
            rows["variable"].append(var); rows["value"].append(val)
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    return WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                         model="best_match", frame=frame)


def test_forecast_matrix_aligns_to_feature_names(db):
    issue = dt.datetime(2024, 6, 1, 0, 0)
    write_weather_series(db, _fc_series(9, issue, 24))
    names = ["wind_speed_100m", "wind_speed_cube", "MISSING_FEATURE"]
    X, vts = forecast_matrix(db, point_id=9, kind="wind", feature_blocks=["wind_power"],
                             feature_names=names, issue_time=issue, horizon_hours=24)
    assert X.shape == (1, 3)            # 1 row (valid=issue+24h), 3 cols (names order)
    assert X[0, 2] == 0.0               # MISSING_FEATURE → 0
    assert vts[0] == issue + dt.timedelta(hours=24)


# --- B2-t5: multipoint / multi-source serving parity ---

def _prev_multipoint(db, plant_id, point_vals, n=40):
    """PREVIOUS_RUNS grid weather + production. point_vals: {pid: wind_100m constant}."""
    create_plant(db, Plant(plant_id=plant_id, name="G", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    for pid, w100 in point_vals.items():
        rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
        for vt in vts:
            for var, val in [("wind_speed_100m", w100), ("wind_speed_10m", w100 * 0.6),
                             ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
                rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
                rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
        frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"),
                                                     "issue_time": pl.Datetime("us"),
                                                     "lead_hours": pl.Int32, "value": pl.Float64})
        write_weather_series(db, WeatherSeries(point_id=pid, role=ApiRole.PREVIOUS_RUNS.value,
                                               model="best_match", frame=frame))
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(1 + (i % 5)) for i in range(n)]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, plant_id, prod)
    return vts


def _fc_multipoint(db, point_vals, issue, horizon, n=24):
    """PREVIOUS_RUNS grid weather with the SAME per-point values as training."""
    vts = [issue + dt.timedelta(hours=i) for i in range(1, n + 1)]
    for pid, w100 in point_vals.items():
        rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
        for vt in vts:
            for var, val in [("wind_speed_100m", w100), ("wind_speed_10m", w100 * 0.6),
                             ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
                rows["valid_time"].append(vt)
                rows["issue_time"].append(vt - dt.timedelta(hours=horizon))
                rows["lead_hours"].append(horizon)
                rows["variable"].append(var); rows["value"].append(float(val))
        frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"),
                                                     "issue_time": pl.Datetime("us"),
                                                     "lead_hours": pl.Int32, "value": pl.Float64})
        write_weather_series(db, WeatherSeries(point_id=pid, role=ApiRole.PREVIOUS_RUNS.value,
                                               model="best_match", frame=frame))


def test_forecast_matrix_multipoint_matches_training_featurization(db):
    point_vals = {7: 7.0, 8: 9.0}  # spatial_mean → 8.0
    vts = _prev_multipoint(db, "grid1", point_vals)
    cfg = FeatureConfig(blocks=["wind_power", "air_density"])
    raw = build_multipoint_dataset(db, plant_id="grid1", point_ids=[7, 8], horizon_hours=24)
    ds = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0,
                   point_policy="spatial_mean", point_ids=[7, 8])

    # Serve for a valid_time present in the training universe.
    target_vt = vts[30]
    issue = target_vt - dt.timedelta(hours=24)
    _fc_multipoint(db, point_vals, issue, 24, n=24)
    X, vt = forecast_matrix_multipoint(
        db, point_ids=[7, 8], kind="wind", feature_blocks=cfg.blocks,
        feature_names=ds.feature_names, issue_time=issue, horizon_hours=24,
        point_policy="spatial_mean", source_policy=None,
    )
    assert vt.shape == (1,)
    assert vt[0] == np.datetime64(target_vt)
    # Row from the training featurization at the same valid_time.
    idx = int(np.where(ds.valid_time == np.datetime64(target_vt))[0][0])
    assert np.allclose(X[0], ds.X[idx]), "served vector must equal training featurization"


def test_forecast_matrix_multipoint_missing_point_degrades_explicitly(db):
    point_vals = {7: 7.0, 8: 9.0}
    _prev_multipoint(db, "grid1", point_vals)
    issue = dt.datetime(2024, 2, 1, 0, 0)
    # Only point 7 gets previous-runs weather (through the target lead) — grid incomplete.
    _fc_multipoint(db, {7: 7.0}, issue, 24, n=24)
    X, vt = forecast_matrix_multipoint(
        db, point_ids=[7, 8], kind="wind", feature_blocks=["wind_power"],
        feature_names=["wind_speed_100m", "wind_speed_cube"],
        issue_time=issue, horizon_hours=24, point_policy="spatial_mean", source_policy=None,
    )
    # Inner join over points → no shared rows for the missing point → empty (explicit, not silent).
    assert X.shape == (0, 2)
    assert vt.shape == (0,)


def test_predict_serving_multipoint_routes_through_policy_transform(db):
    from openenergy.experiments.strategy import FittedCandidate, StrategyArtifact
    from openenergy.experiments.search_space import PipelineConfig
    from openenergy.models.linear_model import RidgeModel

    point_vals = {7: 7.0, 8: 9.0}
    vts = _prev_multipoint(db, "grid1", point_vals)
    cfg_blocks = ["wind_power", "air_density"]
    raw = build_multipoint_dataset(db, plant_id="grid1", point_ids=[7, 8], horizon_hours=24)
    ds = featurize(raw, kind="wind", feature_config=FeatureConfig(blocks=cfg_blocks),
                   capacity_mw=10.0, point_policy="spatial_mean", point_ids=[7, 8])
    model = RidgeModel(alpha=1.0)
    model.fit(ds.X, ds.y)

    pcfg = PipelineConfig(model_family="ridge", nwp_source="best_match", feature_blocks=cfg_blocks,
                          params={"alpha": 1.0}, point_policy="spatial_mean")
    cand = FittedCandidate(config=pcfg, model=model, feature_blocks=cfg_blocks,
                           feature_names=ds.feature_names, cv_nrmse=0.1)
    art = StrategyArtifact(candidates=[cand], ensemble_method="none",
                           point_ids=[7, 8], source_policy=None)

    target_vt = vts[30]
    issue = target_vt - dt.timedelta(hours=24)
    _fc_multipoint(db, point_vals, issue, 24, n=24)
    pred, served_vt = art.predict_serving(db, point_id=7, kind="wind",
                                          issue_time=issue, horizon_hours=24,
                                          capacity_mw=10.0)
    assert served_vt.shape == (1,)
    assert served_vt[0] == np.datetime64(target_vt)
    # predict_serving must reproduce the model's prediction on the training feature row.
    idx = int(np.where(ds.valid_time == np.datetime64(target_vt))[0][0])
    assert np.allclose(pred.p50, model.predict(ds.X[idx:idx + 1]).p50)


def test_predict_serving_ensemble_preserves_monotone_bands(db):
    """D1-t3: a served ensemble keeps non-crossing p10<=p50<=p90 with non-degenerate
    width — bands are combined per-quantile, not dropped."""
    from openenergy.experiments.strategy import FittedCandidate, StrategyArtifact
    from openenergy.experiments.search_space import PipelineConfig
    from openenergy.models.base import ModelWrapper, Prediction

    issue = dt.datetime(2024, 6, 1, 0, 0)
    write_weather_series(db, _fc_series(9, issue, 24, n=24))
    names = ["wind_speed_100m", "wind_speed_cube"]

    class _Banded(ModelWrapper):
        def __init__(self, spread):
            self.spread = spread

        def fit(self, X, y):
            return self

        def predict(self, X):
            p50 = X[:, 0]
            return Prediction(p50=p50, p10=p50 - self.spread, p90=p50 + self.spread)

    def _cand(spread):
        pcfg = PipelineConfig(model_family="lightgbm", nwp_source="best_match",
                              feature_blocks=["wind_power"], params={},
                              point_policy="single_point")
        return FittedCandidate(config=pcfg, model=_Banded(spread),
                               feature_blocks=["wind_power"], feature_names=names, cv_nrmse=0.1)

    art = StrategyArtifact(candidates=[_cand(1.0), _cand(3.0)], ensemble_method="mean")
    pred, vt = art.predict_serving(db, point_id=9, kind="wind", issue_time=issue,
                                   horizon_hours=24, capacity_mw=10.0)
    assert vt.shape == (1,)
    assert pred.p10 is not None and pred.p90 is not None
    assert np.all(pred.p10 <= pred.p50) and np.all(pred.p50 <= pred.p90)
    assert np.all(pred.p90 - pred.p10 > 0.0)  # non-degenerate width


def test_predict_serving_single_point_unchanged(db):
    """point_ids=None keeps serving on the legacy single-point forecast_matrix path."""
    from openenergy.experiments.strategy import FittedCandidate, StrategyArtifact
    from openenergy.experiments.search_space import PipelineConfig
    from openenergy.models.base import ModelWrapper, Prediction

    issue = dt.datetime(2024, 6, 1, 0, 0)
    write_weather_series(db, _fc_series(9, issue, 24, n=24))
    names = ["wind_speed_100m", "wind_speed_cube"]

    class _Echo(ModelWrapper):  # returns the raw feature matrix's first column as p50
        def fit(self, X, y):
            return self

        def predict(self, X):
            return Prediction(p50=X[:, 0])

    pcfg = PipelineConfig(model_family="ridge", nwp_source="best_match",
                          feature_blocks=["wind_power"], params={}, point_policy="single_point")
    cand = FittedCandidate(config=pcfg, model=_Echo(), feature_blocks=["wind_power"],
                           feature_names=names, cv_nrmse=0.1)
    art = StrategyArtifact(candidates=[cand], ensemble_method="none")  # point_ids=None
    X_ref, _ = forecast_matrix(db, point_id=9, kind="wind", feature_blocks=["wind_power"],
                               feature_names=names, issue_time=issue, horizon_hours=24)
    pred, vt = art.predict_serving(db, point_id=9, kind="wind", issue_time=issue,
                                   horizon_hours=24, capacity_mw=10.0)
    assert vt.shape == (1,)
    assert np.allclose(pred.p50, X_ref[:, 0])
