"""In-sample backtest of the champion strategy on its held-out test window.

An experiment reports a champion + metrics but does not persist the actual
test-window predictions. This reconstructs them: it loads the champion strategy
artifact (whose candidate models are already fitted on the train window),
rebuilds the exact raw frame + walk-forward split the runner used, predicts the
top-K candidates on the test indices, combines them through the artifact's
ensemble (bands + conformal calibration included), and returns a time series of
actual vs p10/p50/p90 in MW — so the model's forecast can be charted against
reality over time.

Wind (capacity_norm) only for now: predictions are de-normalised as p·capacity.
Grid champions (artifact.point_ids set) are fully supported; a single-point
champion needs its point_id passed in.
"""
from __future__ import annotations

import datetime as dt

import duckdb

from openenergy.datasets.horizon import build_multipoint_dataset
from openenergy.evaluation.splits import walk_forward_split
from openenergy.experiments.assembly import assemble_raw, featurize
from openenergy.experiments.persistence import get_champion_strategy_model
from openenergy.experiments.strategy import StrategyArtifact
from openenergy.features import FeatureConfig
from openenergy.forecasting.predict import generate_forecast


def backtest_champion(con: duckdb.DuckDBPyConnection, *, plant_id: str, horizon_hours: int,
                      capacity_mw: float, kind: str, point_id: int | None = None,
                      test_size: int | None = None) -> list[dict]:
    """Return the champion's test-window backtest: [{valid_time, actual, p10, p50, p90}] in MW.

    Empty list when there is no champion strategy or too little data.
    """
    champ = get_champion_strategy_model(con, plant_id, horizon_hours)
    if champ is None or not champ.get("artifact_path"):
        return []
    art = StrategyArtifact.load(champ["artifact_path"])

    if art.point_ids:
        raw = build_multipoint_dataset(con, plant_id=plant_id, point_ids=art.point_ids,
                                       horizon_hours=horizon_hours, source_policy=art.source_policy)
    elif point_id is not None:
        raw = assemble_raw(con, plant_id=plant_id, point_id=point_id, horizon_hours=horizon_hours)
    else:
        return []

    n = raw.height
    if n < 30 or not art.candidates:
        return []
    ts = test_size or max(1, n // 5)
    _, te_idx = walk_forward_split(n, test_size=ts)

    p50s: list = []
    p10s: list = []
    p90s: list = []
    valid_time = None
    actual = None
    for cand in art.candidates:
        ds = featurize(
            raw, kind=kind, feature_config=FeatureConfig(blocks=cand.feature_blocks),
            capacity_mw=capacity_mw,
            point_policy=getattr(cand.config, "point_policy", "single_point"),
            point_ids=art.point_ids,
        )
        pred = cand.model.predict(ds.X[te_idx])
        p50s.append(pred.p50)
        if pred.p10 is not None:
            p10s.append(pred.p10)
        if pred.p90 is not None:
            p90s.append(pred.p90)
        valid_time = ds.valid_time[te_idx]
        actual = ds.y[te_idx]

    combined = art._combine(p50s, p10s, p90s)

    def mw(arr, i):
        return max(0.0, float(arr[i]) * capacity_mw)

    series: list[dict] = []
    for i in range(len(valid_time)):
        series.append({
            "valid_time": str(valid_time[i]),
            "actual": max(0.0, float(actual[i]) * capacity_mw),
            "p50": mw(combined.p50, i),
            "p10": mw(combined.p10, i) if combined.p10 is not None else None,
            "p90": mw(combined.p90, i) if combined.p90 is not None else None,
        })
    return series


def backfill_forecasts(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    point_id: int,
    horizons: list[int],
    capacity_mw: float,
    kind: str,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    lookback_days: int = 92,
) -> dict:
    """Persist historical champion forecasts from the previous-runs archive.

    This is a hindcast of the production serving path, not the held-out backtest
    above: every eligible historical target is passed through
    :func:`generate_forecast`, so it uses the pointer-selected champion and the
    exact same previous-runs feature assembly as forward serving.

    Eligibility requires both horizon-matched previous-runs weather and realised
    production.  Existing forecast natural keys are deliberately skipped rather
    than updated: a later backfill must never rewrite a genuine forward-serving
    row.  Consequently replaying the same request is idempotent.
    """
    requested = sorted(set(horizons))
    if not requested:
        raise ValueError("horizons boş olamaz")
    invalid = [h for h in requested if h <= 0 or h % 24 != 0]
    if invalid:
        raise ValueError(
            "previous-runs backfill yalnız pozitif 24 saat katlarını destekler: "
            f"{invalid}"
        )
    if lookback_days <= 0:
        raise ValueError("lookback_days pozitif olmalı")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start_time end_time'dan sonra olamaz")

    # Resolve every horizon through the serving registry before writing anything.
    # This avoids a partial backfill when one requested horizon has no deployed
    # champion and records the exact lineage used by this run.
    champion_ids: dict[int, int] = {}
    for horizon in requested:
        champ = get_champion_strategy_model(con, plant_id, horizon)
        if champ is None:
            raise ValueError(
                f"No champion strategy found for plant_id={plant_id!r}, "
                f"horizon_hours={horizon}"
            )
        champion_ids[horizon] = int(champ["strategy_trial_id"])

    rows_by_horizon: dict[int, int] = {}
    skipped_by_horizon: dict[int, int] = {}
    eligible_by_horizon: dict[int, int] = {}
    pairs_by_horizon: dict[int, int] = {}

    for horizon in requested:
        bounds = con.execute(
            """
            SELECT max(w.valid_time)
            FROM weather_raw w
            JOIN production p
              ON p.plant_id = ? AND p.ts = w.valid_time
            WHERE w.point_id = ? AND w.role = 'previous_runs'
              AND w.lead_hours = ?
              AND (? IS NULL OR w.valid_time <= ?)
            """,
            [plant_id, point_id, horizon, end_time, end_time],
        ).fetchone()
        latest = bounds[0] if bounds is not None else None
        if latest is None:
            rows_by_horizon[horizon] = 0
            skipped_by_horizon[horizon] = 0
            eligible_by_horizon[horizon] = 0
            pairs_by_horizon[horizon] = 0
            continue

        lower = start_time if start_time is not None else latest - dt.timedelta(days=lookback_days)
        upper = end_time if end_time is not None else latest
        valid_times = [
            row[0]
            for row in con.execute(
                """
                SELECT DISTINCT w.valid_time
                FROM weather_raw w
                JOIN production p
                  ON p.plant_id = ? AND p.ts = w.valid_time
                WHERE w.point_id = ? AND w.role = 'previous_runs'
                  AND w.lead_hours = ?
                  AND w.valid_time >= ? AND w.valid_time <= ?
                ORDER BY w.valid_time
                """,
                [plant_id, point_id, horizon, lower, upper],
            ).fetchall()
        ]

        written = 0
        skipped = 0
        for valid_time in valid_times:
            issue_time = valid_time - dt.timedelta(hours=horizon)
            exists = con.execute(
                """
                SELECT 1
                FROM forecasts
                WHERE plant_id = ? AND point_id = ? AND horizon_hours = ?
                  AND issue_time = ? AND valid_time = ?
                LIMIT 1
                """,
                [plant_id, point_id, horizon, issue_time, valid_time],
            ).fetchone()
            if exists is not None:
                skipped += 1
                continue
            written += generate_forecast(
                con,
                plant_id=plant_id,
                point_id=point_id,
                horizon_hours=horizon,
                capacity_mw=capacity_mw,
                kind=kind,
                issue_time=issue_time,
            )

        pair_count = con.execute(
            """
            SELECT count(*)
            FROM forecasts f
            JOIN production p
              ON p.plant_id = f.plant_id AND p.ts = f.valid_time
            WHERE f.plant_id = ? AND f.point_id = ? AND f.horizon_hours = ?
              AND f.valid_time >= ? AND f.valid_time <= ?
            """,
            [plant_id, point_id, horizon, lower, upper],
        ).fetchone()[0]
        rows_by_horizon[horizon] = written
        skipped_by_horizon[horizon] = skipped
        eligible_by_horizon[horizon] = len(valid_times)
        pairs_by_horizon[horizon] = int(pair_count)

    return {
        "plant_id": plant_id,
        "point_id": point_id,
        "horizons": requested,
        "rows": sum(rows_by_horizon.values()),
        "rows_by_horizon": rows_by_horizon,
        "skipped": sum(skipped_by_horizon.values()),
        "skipped_by_horizon": skipped_by_horizon,
        "eligible_by_horizon": eligible_by_horizon,
        "actual_pairs": sum(pairs_by_horizon.values()),
        "actual_pairs_by_horizon": pairs_by_horizon,
        "strategy_trial_ids": champion_ids,
    }
