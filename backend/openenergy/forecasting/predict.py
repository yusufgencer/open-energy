from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np

from openenergy.assets.repository import PlantPoint
from openenergy.experiments.persistence import (
    get_champion_feature_blocks,
    get_champion_model,
    get_champion_strategy_model,
)
from openenergy.experiments.strategy import StrategyArtifact
from openenergy.forecasting.assembly import forecast_matrix
from openenergy.ingestion.weather import ingest_serving_previous_runs
from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order
from openenergy.physics.solar import daylight_mask
from openenergy.providers.base import WeatherProvider
from openenergy.retraining.postprocessor import apply_postprocessor
from openenergy.calibration.online import record_issued_band, refresh_and_apply


def generate_forecast(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                      horizon_hours: int, capacity_mw: float, kind: str,
                      issue_time: dt.datetime) -> int:
    """Generate forecast using the champion model for plant/horizon and persist to forecasts table.

    Steps:
    1. Load champion model metadata (raises ValueError if none).
    2. Load ModelWrapper from artifact_path.
    3. Get champion's feature_blocks from experiment_trials.
    4. Build feature matrix via forecast_matrix (leakage-safe: FORECAST role, horizon-matched valid_time).
    5. Predict p10/p50/p90.
    6. De-normalize: p * capacity_mw, clip >= 0.
    7. Write to forecasts table. Return row count written.
    """
    # geometry is kept for the solar night-zero mask. predict_serving returns
    # predictions already in normalized-power space (each candidate's kpv ŷ' is
    # inverted through P_cs before combination, T-01), so de-normalisation below
    # is always ŷ × capacity — no target-policy branch on the serving side.
    geometry = None

    strategy_champ = get_champion_strategy_model(con, plant_id, horizon_hours)
    strategy_trial_id = (
        int(strategy_champ["strategy_trial_id"]) if strategy_champ is not None else None
    )
    if strategy_champ is not None and strategy_champ["artifact_path"]:
        artifact = StrategyArtifact.load(strategy_champ["artifact_path"])
        pred, valid_times = artifact.predict_serving(
            con,
            point_id=point_id,
            kind=kind,
            issue_time=issue_time,
            horizon_hours=horizon_hours,
            capacity_mw=capacity_mw,
        )
        geometry = getattr(artifact, "geometry", None)
        model_id = None
    else:
        # Legacy single-model artifacts are still supported, but their model and
        # feature metadata are both resolved through champion_pointer.  Do not
        # consult this fallback when a strategy artifact served successfully:
        # optimistic model flags from a later challenger must not veto or replace
        # the pointer-selected incumbent (T-05). Its lineage remains the
        # pointer's strategy_trial_id even when the legacy model artifact is the
        # executable fallback. With no pointer there is no unambiguous lineage,
        # so serving fails instead of inventing a provenance identifier.
        champ = get_champion_model(con, plant_id, horizon_hours)
        if champ is None:
            raise ValueError(
                f"No champion model found for plant_id={plant_id!r}, "
                f"horizon_hours={horizon_hours}"
            )
        model: ModelWrapper = ModelWrapper.load(champ["artifact_path"])
        feature_blocks = get_champion_feature_blocks(con, plant_id, horizon_hours)
        feature_names: list[str] = champ["feature_names"]

        X, valid_times = forecast_matrix(
            con, point_id=point_id, kind=kind,
            feature_blocks=feature_blocks, feature_names=feature_names,
            issue_time=issue_time, horizon_hours=horizon_hours,
        )

        if X.shape[0] == 0:
            return 0

        pred = model.predict(X)
        model_id = champ["model_id"]

    if len(valid_times) == 0:
        return 0

    # De-normalize to MW: ŷ × capacity, clip >= 0. Predictions arrive in
    # normalized-power space from both the strategy path (kpv already inverted
    # per-candidate inside predict_serving) and the plain champion path, so a
    # single de-normalisation is correct for every target policy (T-01).
    pred = enforce_quantile_order(
        Prediction(
            p50=np.clip(pred.p50 * capacity_mw, 0, None),
            p10=np.clip(pred.p10 * capacity_mw, 0, None) if pred.p10 is not None else None,
            p90=np.clip(pred.p90 * capacity_mw, 0, None) if pred.p90 is not None else None,
        )
    )
    # Daily LASSO post-processor (E1-t3): correct cheap systematic drift/bias on
    # top of the champion forecast, in MW space, using coefficients fitted from
    # recent (forecast, actual) pairs for this plant+horizon. Identity (no-op,
    # byte-identical) when nothing has been fitted, so the legacy path is
    # untouched. Applied before the hard night-zero so that guarantee stays final.
    pred = apply_postprocessor(con, plant_id=plant_id, horizon_hours=horizon_hours, pred=pred)
    # T-21 terminal late-conformal layer: refresh plant+horizon state using only
    # realised valid_times strictly before this issue, then adapt the new band.
    # Keep the postprocessed pre-online band for auditable feedback lineage.
    online_base = pred
    pred, online_state_version = refresh_and_apply(
        con,
        plant_id=plant_id,
        horizon_hours=horizon_hours,
        issue_time=issue_time,
        pred=pred,
    )
    p50, p10, p90 = pred.p50, pred.p10, pred.p90

    # Hard night zero for solar (C1-t4): a solar plant produces exactly 0 at
    # night, so force every quantile to 0 on below-horizon rows regardless of
    # target policy. lat/lon come from the strategy geometry when present, else
    # from the plant_point. Wind is unaffected. kPV already zeroes night via
    # P_cs=0; this makes the guarantee hold on the capacity_norm path too.
    hard_night = np.zeros(len(valid_times), dtype=bool)
    if kind == "solar":
        latlon = None
        if geometry is not None:
            latlon = (geometry.lat, geometry.lon)
        else:
            row = con.execute(
                "SELECT latitude, longitude FROM plant_points WHERE point_id=?",
                [point_id],
            ).fetchone()
            if row is not None:
                latlon = (row[0], row[1])
        if latlon is not None:
            night = ~daylight_mask(valid_times, latlon[0], latlon[1])
            hard_night = np.asarray(night, dtype=bool)
            p50 = np.where(night, 0.0, p50)
            if p10 is not None:
                p10 = np.where(night, 0.0, p10)
            if p90 is not None:
                p90 = np.where(night, 0.0, p90)

    rows_written = 0
    for i in range(len(valid_times)):
        vt = valid_times[i].item()  # numpy datetime64 → Python datetime
        p10_val = float(p10[i]) if p10 is not None else None
        p90_val = float(p90[i]) if p90 is not None else None
        con.execute(
            """INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, valid_time,
               p10, p50, p90, model_id, strategy_trial_id) VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (plant_id, point_id, horizon_hours, issue_time, valid_time)
               DO UPDATE SET p10=excluded.p10, p50=excluded.p50, p90=excluded.p90,
                             model_id=excluded.model_id,
                             strategy_trial_id=excluded.strategy_trial_id""",
            [plant_id, point_id, horizon_hours, issue_time, vt,
             p10_val, float(p50[i]), p90_val, model_id, strategy_trial_id],
        )
        # Night rows are governed by the hard physical zero, not a statistical
        # band, and must neither train nor be widened by online conformal.
        is_hard_night = bool(hard_night[i])
        if (
            not is_hard_night
            and online_base.p10 is not None
            and online_base.p90 is not None
            and p10_val is not None
            and p90_val is not None
        ):
            record_issued_band(
                con,
                plant_id=plant_id,
                horizon_hours=horizon_hours,
                valid_time=vt,
                issue_time=issue_time,
                point_id=point_id,
                base_lower=float(online_base.p10[i]),
                base_upper=float(online_base.p90[i]),
                served_lower=p10_val,
                served_upper=p90_val,
                model_id=model_id,
                strategy_trial_id=strategy_trial_id,
                state_version=online_state_version,
            )
        rows_written += 1

    return rows_written


def run_serving_cycle(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    point: PlantPoint,
    horizons: list[int],
    capacity_mw: float,
    kind: str,
    issue_time: dt.datetime,
    models: list[str] | None = None,
    provider: WeatherProvider | None = None,
) -> dict:
    """Ingest a bounded missing slice and serve one target per horizon.

    The T-13 previous-runs ingestion boundary is invoked at most once per point
    and cycle, only if at least one ``issue_time + horizon`` row is absent.
    Replaying a cycle is safe because ``generate_forecast`` upserts its natural
    issue/valid/horizon key.
    """
    if point.point_id is None:
        raise ValueError("point_id gerekli")
    requested = sorted(set(horizons))
    if not requested:
        raise ValueError("horizons boş olamaz")
    invalid = [h for h in requested if h <= 0 or h % 24 != 0]
    if invalid:
        raise ValueError(
            "serving cycle yalnız pozitif 24 saat katlarını destekler: "
            f"{invalid}"
        )

    missing: list[int] = []
    for horizon in requested:
        target = issue_time + dt.timedelta(hours=horizon)
        found = con.execute(
            """SELECT 1 FROM weather_raw
               WHERE point_id=? AND role='previous_runs' AND lead_hours=?
                 AND valid_time=?
               LIMIT 1""",
            [point.point_id, horizon, target],
        ).fetchone()
        if found is None:
            missing.append(horizon)

    ingest_result = None
    if missing:
        ingest_result = ingest_serving_previous_runs(
            con,
            point,
            kind,
            horizon_hours_list=missing,
            models=models,
            provider=provider,
        )

    rows_by_horizon: dict[int, int] = {}
    for horizon in requested:
        rows_by_horizon[horizon] = generate_forecast(
            con,
            plant_id=plant_id,
            point_id=point.point_id,
            horizon_hours=horizon,
            capacity_mw=capacity_mw,
            kind=kind,
            issue_time=issue_time,
        )
    return {
        "plant_id": plant_id,
        "point_id": point.point_id,
        "issue_time": issue_time,
        "horizons": requested,
        "rows": sum(rows_by_horizon.values()),
        "rows_by_horizon": rows_by_horizon,
        "ingest": ingest_result,
    }
