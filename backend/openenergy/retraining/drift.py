"""E1-t4 — Drift monitor + drift-triggered emergency retrain.

Between the (monthly) scheduled base retrains (E1-t2) a champion's real-world skill
can silently decay — a turbine derates, a panel string soils, the local climate
shifts. This module tracks the *rolling recent error* of the served p50 forecasts
per plant+horizon against the champion's proven test-time skill and, when that
error breaches a threshold **persistently**, enqueues an emergency retrain instead
of waiting for the next monthly cycle.

Two design guarantees:

* **Persistence** — a single noisy window must not trigger a retrain. A breach only
  *arms* a counter; the emergency retrain fires once ``min_consecutive`` consecutive
  evaluations breach the threshold.

* **Hysteresis (fires once)** — after firing, the state latches (``fired=true``) and
  will not enqueue again while the plant remains degraded. It only re-arms once the
  recent error falls back below a *lower* recovery threshold, so a plant hovering
  around the breach line does not thrash the retrain queue.

Recent error is computed from the same recent (forecast, actual) pairs the daily
post-processor uses (E1-t3): for each realised ``valid_time`` we take the
latest-issued p50 forecast at that horizon and join it to realised production, then
normalise RMSE by the plant capacity so it is directly comparable to the champion's
stored ``test_nrmse`` (capacity-normalised).
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Callable

import duckdb
import numpy as np
from scipy.stats import ks_2samp

DEFAULT_MIN_SAMPLES = 12
DEFAULT_LOOKBACK_DAYS = 30
# Recent nRMSE above ``champion_nrmse * BREACH_RATIO`` counts as degraded.
DEFAULT_BREACH_RATIO = 1.5
# Recent nRMSE back below ``champion_nrmse * RECOVER_RATIO`` counts as recovered.
DEFAULT_RECOVER_RATIO = 1.2
# Consecutive breaching evaluations required before the emergency retrain fires.
DEFAULT_MIN_CONSECUTIVE = 2
DEFAULT_PSI_THRESHOLD = 0.25
DEFAULT_KS_ALPHA = 0.05
DEFAULT_FRESHNESS_HOURS = 36.0
DEFAULT_PSI_BINS = 10

_TABLE = "drift_state"

# enqueue callback: (plant_id, horizon_hours) -> job_id
EnqueueFn = Callable[[str, int], int]


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def population_stability_index(
    reference,
    current,
    *,
    bins: int = DEFAULT_PSI_BINS,
    smoothing: float = 1e-6,
) -> float | None:
    """Robust PSI using reference-quantile bins and additive smoothing.

    NaN/inf values are ignored. Degenerate reference distributions get a small
    symmetric bin around their constant value, so an identical constant series
    scores zero while a moved constant series is still detected.
    """
    ref = _finite(reference)
    cur = _finite(current)
    if ref.size == 0 or cur.size == 0:
        return None
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size == 1:
        centre = float(edges[0])
        scale = max(abs(centre), 1.0) * 1e-9
        edges = np.array([-np.inf, centre - scale, centre + scale, np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float) + smoothing
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float) + smoothing
    ref_share = ref_counts / ref_counts.sum()
    cur_share = cur_counts / cur_counts.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


# Short alias for callers/tests using the standard detector name.
psi = population_stability_index


def two_sample_ks(reference, current) -> tuple[float | None, float | None]:
    """Return the two-sided two-sample KS statistic and p-value.

    Non-finite observations are excluded. Empty samples are explicitly
    unjudgeable instead of leaking NaNs into persisted detector state.
    """
    ref = _finite(reference)
    cur = _finite(current)
    if ref.size == 0 or cur.size == 0:
        return None, None
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def _champion_covariates(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    horizon_hours: int,
    fallback_sources: list[str] | None,
) -> tuple[list[str], set[str] | None, dict]:
    """Resolve source/feature metadata through the served champion pointer."""
    rows = con.execute(
        """
        SELECT sc.candidate_config, sc.feature_names, cp.promoted_at,
               cp.strategy_trial_id, st.experiment_id
        FROM champion_pointer cp
        JOIN strategy_trials st ON st.strategy_trial_id=cp.strategy_trial_id
        LEFT JOIN strategy_candidates sc
          ON sc.strategy_trial_id=cp.strategy_trial_id
        WHERE cp.plant_id=? AND cp.horizon_hours=?
        ORDER BY sc.rank
        """,
        [plant_id, horizon_hours],
    ).fetchall()
    sources: list[str] = []
    feature_names: set[str] = set()
    metadata: dict = {"reference_kind": "served_champion_training_assembly"}
    for config_json, names_json, promoted_at, strategy_trial_id, experiment_id in rows:
        cfg = json.loads(config_json) if config_json else {}
        policy = cfg.get("source_policy")
        if not policy or policy == "single_source":
            policy = cfg.get("nwp_source")
        if policy:
            sources.extend(str(policy).split("+"))
        if names_json:
            feature_names.update(json.loads(names_json))
        metadata.update({
            "strategy_trial_id": strategy_trial_id,
            "experiment_id": experiment_id,
            "promoted_at": str(promoted_at) if promoted_at is not None else None,
        })
    if not sources:
        sources = list(fallback_sources or [])
        metadata["reference_kind"] = "historical_assembly_fallback"
    return list(dict.fromkeys(sources)), feature_names or None, metadata


def _persist_drift_event(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    source: str,
    feature_name: str,
    window_end,
    status: str,
    psi_value: float | None = None,
    psi_threshold: float | None = None,
    ks_statistic: float | None = None,
    ks_pvalue: float | None = None,
    ks_alpha: float | None = None,
    reference_start=None,
    reference_end=None,
    current_start=None,
    current_end=None,
    reference_samples: int = 0,
    current_samples: int = 0,
    reference_season: str | None = None,
    freshness_hours: float | None = None,
    freshness_limit_hours: float | None = None,
    metadata: dict | None = None,
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO drift_events
          (plant_id, horizon_hours, source, feature_name, window_end,
           detector_status, psi, psi_threshold, ks_statistic, ks_pvalue, ks_alpha,
           reference_start, reference_end, current_start, current_end,
           reference_samples, current_samples, reference_season,
           freshness_hours, freshness_limit_hours, metadata, evaluated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,now())
        """,
        [
            plant_id, horizon_hours, source, feature_name, window_end, status,
            psi_value, psi_threshold, ks_statistic, ks_pvalue, ks_alpha,
            reference_start, reference_end, current_start, current_end,
            reference_samples, current_samples, reference_season,
            freshness_hours, freshness_limit_hours,
            json.dumps(metadata or {}, sort_keys=True, default=str),
        ],
    )


def assess_nwp_covariate_drift(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    point_id: int,
    horizon_hours: int,
    sources: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    freshness_hours: float = DEFAULT_FRESHNESS_HOURS,
    as_of: dt.datetime | None = None,
) -> dict:
    """Evaluate NWP freshness plus season-matched per-feature PSI/KS.

    Reference and current rows use the same point, source, role, lead and
    variable. The reference is historical assembly data before the recent
    window, first matched by month and then (when necessary) by quarter.
    """
    required_sources, champion_features, champion_meta = _champion_covariates(
        con, plant_id, horizon_hours, sources
    )
    if not required_sources:
        required_sources = list(sources or [])
    now = as_of or dt.datetime.now()
    results: list[dict] = []
    overall = "ok"

    for source in required_sources:
        latest = con.execute(
            """
            SELECT max(issue_time), max(valid_time)
            FROM weather_raw
            WHERE point_id=? AND role='previous_runs' AND lead_hours=? AND model=?
            """,
            [point_id, horizon_hours, source],
        ).fetchone()
        latest_issue, latest_valid = latest if latest is not None else (None, None)
        window_key = latest_valid or now
        if latest_issue is None:
            freshness_status = "missing"
            age_hours = None
            overall = "nwp_missing"
        else:
            age_hours = max(0.0, (now - latest_issue).total_seconds() / 3600.0)
            freshness_status = "stale" if age_hours > freshness_hours else "ok"
            if freshness_status == "stale" and overall != "nwp_missing":
                overall = "nwp_stale"
        _persist_drift_event(
            con, plant_id=plant_id, horizon_hours=horizon_hours, source=source,
            feature_name="__freshness__", window_end=window_key,
            status=freshness_status, current_end=latest_valid,
            freshness_hours=age_hours, freshness_limit_hours=freshness_hours,
            metadata=champion_meta,
        )
        results.append({
            "source": source, "feature_name": "__freshness__",
            "status": freshness_status, "freshness_hours": age_hours,
            "freshness_limit_hours": freshness_hours,
        })
        if latest_valid is None:
            continue

        current_start = latest_valid - dt.timedelta(days=lookback_days)
        variables = [
            r[0] for r in con.execute(
                """
                SELECT DISTINCT variable FROM weather_raw
                WHERE point_id=? AND role='previous_runs' AND lead_hours=? AND model=?
                ORDER BY variable
                """,
                [point_id, horizon_hours, source],
            ).fetchall()
        ]
        if champion_features is not None:
            direct = [v for v in variables if v in champion_features]
            if direct:
                variables = direct

        for variable in variables:
            rows = con.execute(
                """
                SELECT valid_time, value FROM weather_raw
                WHERE point_id=? AND role='previous_runs' AND lead_hours=?
                  AND model=? AND variable=? AND value IS NOT NULL
                ORDER BY valid_time
                """,
                [point_id, horizon_hours, source, variable],
            ).fetchall()
            current_rows = [(ts, value) for ts, value in rows if ts >= current_start]
            historical = [(ts, value) for ts, value in rows if ts < current_start]
            month_ref = [(ts, value) for ts, value in historical
                         if ts.month == latest_valid.month]
            quarter = (latest_valid.month - 1) // 3
            quarter_ref = [(ts, value) for ts, value in historical
                           if (ts.month - 1) // 3 == quarter]
            if len(month_ref) >= min_samples:
                reference_rows, season = month_ref, f"month:{latest_valid.month:02d}"
            elif len(quarter_ref) >= min_samples:
                reference_rows, season = quarter_ref, f"quarter:{quarter + 1}"
            else:
                reference_rows, season = historical, "all_history"

            ref = [value for _, value in reference_rows]
            cur = [value for _, value in current_rows]
            if len(_finite(ref)) < min_samples or len(_finite(cur)) < min_samples:
                status, psi_value, ks_stat, ks_p = "insufficient_data", None, None, None
                if overall == "ok":
                    overall = "insufficient_data"
            else:
                psi_value = population_stability_index(ref, cur)
                ks_stat, ks_p = two_sample_ks(ref, cur)
                drifted = (
                    psi_value is not None and psi_value > psi_threshold
                ) or (
                    ks_p is not None and ks_p < ks_alpha
                )
                status = "drift" if drifted else "ok"
                if drifted and overall == "ok":
                    overall = "covariate_drift"
            ref_start = reference_rows[0][0] if reference_rows else None
            ref_end = reference_rows[-1][0] if reference_rows else None
            _persist_drift_event(
                con, plant_id=plant_id, horizon_hours=horizon_hours, source=source,
                feature_name=variable, window_end=latest_valid, status=status,
                psi_value=psi_value, psi_threshold=psi_threshold,
                ks_statistic=ks_stat, ks_pvalue=ks_p, ks_alpha=ks_alpha,
                reference_start=ref_start, reference_end=ref_end,
                current_start=current_start, current_end=latest_valid,
                reference_samples=len(_finite(ref)), current_samples=len(_finite(cur)),
                reference_season=season, metadata=champion_meta,
            )
            results.append({
                "source": source, "feature_name": variable, "status": status,
                "psi": psi_value, "psi_threshold": psi_threshold,
                "ks_statistic": ks_stat, "ks_pvalue": ks_p, "ks_alpha": ks_alpha,
                "reference_samples": len(_finite(ref)),
                "current_samples": len(_finite(cur)),
                "reference_season": season,
            })
    return {"status": overall, "features": results}


def _recent_p50_pairs(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(forecast_p50, actual)`` arrays from recent (forecast, actual) pairs.

    For each realised ``valid_time`` the *latest-issued* p50 forecast at this horizon
    is kept and joined to realised production; only pairs within ``lookback_days`` of
    the most recent realised pair are used so stale error expires.
    """
    rows = con.execute(
        """
        SELECT valid_time, p50, power_mw FROM (
            SELECT f.valid_time AS valid_time, f.p50 AS p50, pr.power_mw AS power_mw,
                   row_number() OVER (
                       PARTITION BY f.valid_time
                       ORDER BY f.issue_time DESC, f.created_at DESC
                   ) AS rn
            FROM forecasts f
            JOIN production pr
              ON pr.plant_id = f.plant_id AND pr.ts = f.valid_time
            WHERE f.plant_id = ? AND f.horizon_hours = ?
        ) WHERE rn = 1
        ORDER BY valid_time
        """,
        [plant_id, horizon_hours],
    ).fetchall()

    if not rows:
        return np.array([]), np.array([])

    latest = max(r[0] for r in rows)
    cutoff = latest - dt.timedelta(days=lookback_days)
    fs, ys = [], []
    for vt, p50, actual in rows:
        if vt < cutoff or p50 is None or actual is None:
            continue
        fs.append(float(p50))
        ys.append(float(actual))
    return np.asarray(fs, dtype=float), np.asarray(ys, dtype=float)


def recent_nrmse(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    capacity_mw: float,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> tuple[float | None, int]:
    """Rolling recent capacity-normalised RMSE of served p50 forecasts.

    Returns ``(nrmse, n_samples)``. ``nrmse`` is ``None`` when there are fewer than
    ``min_samples`` recent pairs (too little evidence to judge drift).
    """
    f, y = _recent_p50_pairs(
        con, plant_id=plant_id, horizon_hours=horizon_hours, lookback_days=lookback_days
    )
    n = int(len(y))
    if n < min_samples or capacity_mw <= 0:
        return None, n
    rmse = float(np.sqrt(np.mean((f - y) ** 2)))
    return rmse / float(capacity_mw), n


def _champion_nrmse(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int
) -> float | None:
    # Reference the SERVED strategy champion (champion_pointer → strategy_trials):
    # drift must monitor what is actually forecast, not the offline model-table
    # champion (T-06). Fall back to the experiment champion only before a strategy
    # champion has been promoted onto the serving path.
    row = con.execute(
        """
        SELECT st.test_nrmse
        FROM champion_pointer cp
        JOIN strategy_trials st ON st.strategy_trial_id = cp.strategy_trial_id
        WHERE cp.plant_id = ? AND cp.horizon_hours = ?
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    if row is None or row[0] is None:
        row = con.execute(
            """
            SELECT et.test_nrmse
            FROM experiment_trials et
            JOIN experiments e ON et.experiment_id = e.experiment_id
            WHERE e.plant_id = ? AND et.horizon_hours = ? AND et.is_champion = true
            ORDER BY et.trial_id DESC LIMIT 1
            """,
            [plant_id, horizon_hours],
        ).fetchone()
    if row is None or row[0] is None:
        return None
    val = float(row[0])
    if np.isnan(val):
        return None
    return val


def _latest_pair_vt(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int
):
    """Latest realised (forecast, actual) valid_time — the drift *evaluation window*
    key. It advances only when new realised production arrives, so re-running
    check_drift on unchanged data is a repeat of the same window (T-06)."""
    row = con.execute(
        """
        SELECT max(f.valid_time)
        FROM forecasts f
        JOIN production pr ON pr.plant_id = f.plant_id AND pr.ts = f.valid_time
        WHERE f.plant_id = ? AND f.horizon_hours = ?
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    return row[0] if row is not None else None


def _load_state(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int
) -> dict:
    row = con.execute(
        f"SELECT consecutive_breaches, fired, last_job_id, last_window FROM {_TABLE} "
        f"WHERE plant_id = ? AND horizon_hours = ?",
        [plant_id, horizon_hours],
    ).fetchone()
    if row is None:
        return {"consecutive_breaches": 0, "fired": False, "last_job_id": None,
                "last_window": None}
    return {
        "consecutive_breaches": int(row[0]),
        "fired": bool(row[1]),
        "last_job_id": row[2],
        "last_window": row[3],
    }


def _save_state(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    consecutive: int,
    fired: bool,
    recent: float | None,
    champion: float | None,
    ratio: float | None,
    last_job_id: int | None,
    last_window=None,
) -> None:
    con.execute(f"DELETE FROM {_TABLE} WHERE plant_id = ? AND horizon_hours = ?",
                [plant_id, horizon_hours])
    con.execute(
        f"INSERT INTO {_TABLE} (plant_id, horizon_hours, consecutive_breaches, fired, "
        f"recent_nrmse, champion_nrmse, ratio, last_job_id, last_window, updated_at) "
        f"VALUES (?,?,?,?,?,?,?,?,?, now())",
        [plant_id, horizon_hours, consecutive, fired, recent, champion, ratio,
         last_job_id, last_window],
    )


def check_drift(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    capacity_mw: float,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    breach_ratio: float = DEFAULT_BREACH_RATIO,
    recover_ratio: float = DEFAULT_RECOVER_RATIO,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    enqueue: EnqueueFn | None = None,
    point_id: int | None = None,
    nwp_sources: list[str] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    freshness_hours: float = DEFAULT_FRESHNESS_HOURS,
    as_of: dt.datetime | None = None,
) -> dict:
    """Evaluate drift for one plant+horizon and (once, persistently) enqueue retrain.

    Returns a status dict::

        {plant_id, horizon_hours, recent_nrmse, champion_nrmse, ratio,
         consecutive_breaches, breaching, fired, job_id, status}

    ``fired`` is True only on the evaluation that actually enqueued a retrain this
    call; ``breaching`` is the latched state (retrain already fired, awaiting
    recovery). ``status`` is one of ``no_champion``, ``insufficient_data``, ``ok``,
    ``degraded`` (breaching, arming), ``retrain_enqueued`` (fired this call), or
    ``armed`` (already fired, still degraded).
    """
    state = _load_state(con, plant_id, horizon_hours)
    consecutive = state["consecutive_breaches"]
    latched = state["fired"]
    last_job_id = state["last_job_id"]

    champion = _champion_nrmse(con, plant_id, horizon_hours)
    recent, n = recent_nrmse(
        con, plant_id=plant_id, horizon_hours=horizon_hours, capacity_mw=capacity_mw,
        lookback_days=lookback_days, min_samples=min_samples,
    )
    nwp = (
        assess_nwp_covariate_drift(
            con, plant_id=plant_id, point_id=point_id,
            horizon_hours=horizon_hours, sources=nwp_sources,
            lookback_days=lookback_days, min_samples=min_samples,
            psi_threshold=psi_threshold, ks_alpha=ks_alpha,
            freshness_hours=freshness_hours, as_of=as_of,
        )
        if point_id is not None
        else {"status": "not_checked", "features": []}
    )

    base = {
        "plant_id": plant_id,
        "horizon_hours": horizon_hours,
        "recent_nrmse": recent,
        "champion_nrmse": champion,
        "n_samples": n,
        "consecutive_breaches": consecutive,
        "breaching": latched,
        "fired": False,
        "job_id": None,
        "nwp_status": nwp["status"],
        "detectors": nwp["features"],
    }

    if champion is None:
        return {**base, "ratio": None, "status": "no_champion"}
    if recent is None:
        return {**base, "ratio": None, "status": "insufficient_data"}

    ratio = float("inf") if champion <= 0 else recent / champion
    breached = ratio > breach_ratio
    recovered = ratio < recover_ratio

    # Persistence is over EVALUATION WINDOWS, not calls (T-06). Re-running on the
    # same realised data must not advance the breach counter or fire — otherwise a
    # loop of no-op calls would trip an emergency retrain from a single window.
    window = _latest_pair_vt(con, plant_id, horizon_hours)
    if window is not None and state["last_window"] is not None and window == state["last_window"]:
        status = "armed" if latched else ("degraded" if breached else "ok")
        if status == "ok" and nwp["status"] not in {"ok", "not_checked"}:
            status = nwp["status"]
        return {**base, "ratio": ratio, "status": status}

    fired_now = False
    job_id: int | None = None

    if latched:
        # Already retrained for this degradation episode — only re-arm on recovery.
        if recovered:
            latched = False
            consecutive = 0
    else:
        if breached:
            consecutive += 1
            if consecutive >= min_consecutive:
                if enqueue is not None:
                    job_id = int(enqueue(plant_id, horizon_hours))
                    last_job_id = job_id
                latched = True
                fired_now = True
        else:
            consecutive = 0

    _save_state(
        con, plant_id=plant_id, horizon_hours=horizon_hours, consecutive=consecutive,
        fired=latched, recent=recent, champion=champion, ratio=ratio,
        last_job_id=last_job_id, last_window=window,
    )

    if fired_now:
        status = "retrain_enqueued"
    elif latched:
        status = "armed"
    elif breached:
        status = "degraded"
    else:
        status = "ok"
    if status == "ok" and nwp["status"] not in {"ok", "not_checked"}:
        status = nwp["status"]

    return {
        **base,
        "ratio": ratio,
        "consecutive_breaches": consecutive,
        "breaching": latched,
        "fired": fired_now,
        "job_id": job_id,
        "status": status,
    }


def drift_status(con: duckdb.DuckDBPyConnection, plant_id: str) -> list[dict]:
    """Return the persisted drift state rows for a plant (status surface for the API)."""
    rows = con.execute(
        f"SELECT horizon_hours, consecutive_breaches, fired, recent_nrmse, "
        f"champion_nrmse, ratio, last_job_id, updated_at FROM {_TABLE} "
        f"WHERE plant_id = ? ORDER BY horizon_hours",
        [plant_id],
    ).fetchall()
    result = [
        {
            "horizon_hours": int(r[0]),
            "consecutive_breaches": int(r[1]),
            "breaching": bool(r[2]),
            "recent_nrmse": r[3],
            "champion_nrmse": r[4],
            "ratio": r[5],
            "last_job_id": r[6],
            "updated_at": str(r[7]),
        }
        for r in rows
    ]
    for item in result:
        event_rows = con.execute(
            """
            SELECT source, feature_name, detector_status, psi, psi_threshold,
                   ks_statistic, ks_pvalue, ks_alpha, reference_start,
                   reference_end, current_start, current_end, reference_samples,
                   current_samples, reference_season, freshness_hours,
                   freshness_limit_hours, metadata, evaluated_at
            FROM drift_events
            WHERE plant_id=? AND horizon_hours=?
              AND window_end=(
                  SELECT max(window_end) FROM drift_events
                  WHERE plant_id=? AND horizon_hours=?
              )
            ORDER BY source, feature_name
            """,
            [plant_id, item["horizon_hours"], plant_id, item["horizon_hours"]],
        ).fetchall()
        item["detectors"] = [
            {
                "source": r[0], "feature_name": r[1], "status": r[2],
                "psi": r[3], "psi_threshold": r[4], "ks_statistic": r[5],
                "ks_pvalue": r[6], "ks_alpha": r[7],
                "reference_start": str(r[8]) if r[8] is not None else None,
                "reference_end": str(r[9]) if r[9] is not None else None,
                "current_start": str(r[10]) if r[10] is not None else None,
                "current_end": str(r[11]) if r[11] is not None else None,
                "reference_samples": int(r[12]), "current_samples": int(r[13]),
                "reference_season": r[14], "freshness_hours": r[15],
                "freshness_limit_hours": r[16],
                "metadata": json.loads(r[17]) if r[17] else {},
                "evaluated_at": str(r[18]),
            }
            for r in event_rows
        ]
    return result
