from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import duckdb


@dataclass(frozen=True)
class ScenarioRun:
    experiment_id: int
    plant_id: str
    asset_kind: str
    horizon_hours: int
    strategy_family: str
    strategy_trial_id: int | None = None
    feature_signature: str | None = None
    model_signature: str | None = None
    train_policy: str | None = None
    point_policy: str | None = None
    source_policy: str | None = None
    target_policy: str | None = None
    ensemble_policy: str | None = None
    quantile_policy: str | None = None
    nrmse: float | None = None
    nmae: float | None = None
    bias: float | None = None
    pinball: float | None = None
    coverage: float | None = None
    crps: float | None = None
    peak_bias: float | None = None
    peak_mae: float | None = None
    skill_score: float | None = None
    evaluation_grade: str = "exploratory"
    evaluation_n_days: int | None = None
    evaluation_origins: int | None = None
    evaluation_seasons: int | None = None
    runtime_seconds: float | None = None
    is_champion: bool = False


def feature_signature(feature_blocks: list[str]) -> str:
    return "+".join(sorted(dict.fromkeys(feature_blocks)))


def model_signature(model_families: list[str]) -> str:
    return "+".join(sorted(dict.fromkeys(model_families)))


def strategy_family(
    *,
    asset_kind: str,
    ensemble_policy: str,
    has_nwp_spread: bool = False,
    point_policy: str | None = None,
    source_policy: str | None = None,
    target_policy: str | None = None,
) -> str:
    parts = [asset_kind]
    if has_nwp_spread:
        parts.append("nwp_spread")
    parts.extend(["topk", ensemble_policy])
    # Fold spatial/source regimes into the family so leaderboards distinguish
    # them. Trivial single-point/single-source regimes stay compact for
    # backward compatibility (they add no suffix).
    if point_policy and point_policy != "single_point":
        parts.append(point_policy)
    if source_policy and source_policy != "single_source":
        parts.append(source_policy)
    # Fold the target transform (C1-t7) so kpv vs capacity_norm champions land in
    # distinct families. The legacy capacity_norm target adds no suffix.
    if target_policy and target_policy != "capacity_norm":
        parts.append(target_policy)
    return "_".join(parts)


def pooled_strategy_family(asset_kind: str) -> str:
    """Family tag for the F1-t2 pooled-GBM competitor.

    One model is trained across ALL leads with ``lead_time_hours`` as a feature,
    then judged per lead against the per-horizon champions. Tagging it distinctly
    (``{kind}_pooled_gbm``) keeps it a first-class scenario-memory competitor: the
    leaderboard, grouped by (asset_kind, horizon_hours, strategy_family), then
    shows the pooled family side by side with the per-horizon families at each
    horizon — the pooled-vs-per-horizon verdict.
    """
    return f"{asset_kind}_pooled_gbm"


def record_scenario_run(con: duckdb.DuckDBPyConnection, run: ScenarioRun) -> int:
    return con.execute(
        """
        INSERT INTO scenario_runs
          (experiment_id, plant_id, asset_kind, horizon_hours, strategy_trial_id,
           strategy_family, feature_signature, model_signature, train_policy,
           point_policy, source_policy, target_policy, ensemble_policy, quantile_policy,
           nrmse, nmae, bias,
           pinball, coverage, crps, peak_bias, peak_mae, skill_score,
           evaluation_grade, evaluation_n_days, evaluation_origins, evaluation_seasons,
           runtime_seconds, is_champion)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        RETURNING scenario_run_id
        """,
        [
            run.experiment_id,
            run.plant_id,
            run.asset_kind,
            run.horizon_hours,
            run.strategy_trial_id,
            run.strategy_family,
            run.feature_signature,
            run.model_signature,
            run.train_policy,
            run.point_policy,
            run.source_policy,
            run.target_policy,
            run.ensemble_policy,
            run.quantile_policy,
            run.nrmse,
            run.nmae,
            run.bias,
            run.pinball,
            run.coverage,
            run.crps,
            run.peak_bias,
            run.peak_mae,
            run.skill_score,
            run.evaluation_grade,
            run.evaluation_n_days,
            run.evaluation_origins,
            run.evaluation_seasons,
            run.runtime_seconds,
            run.is_champion,
        ],
    ).fetchone()[0]


def update_scenario_champion(
    con: duckdb.DuckDBPyConnection,
    *,
    experiment_id: int,
    horizon_hours: int,
    is_champion: bool,
) -> None:
    """Reconcile a retrain's scenario row with the operational promotion decision.

    ``run_experiment`` records every new experiment's best strategy as
    ``is_champion=True`` (offline optimism). When the champion-challenger gate then
    *defends* the incumbent, the challenger never actually served — this flips its
    scenario row's ``is_champion`` to False so the leaderboard reflects operational
    reality (E1-t5), not the offline experiment alone.
    """
    con.execute(
        "UPDATE scenario_runs SET is_champion=? WHERE experiment_id=? AND horizon_hours=?",
        [is_champion, experiment_id, horizon_hours],
    )


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(median(present)) if present else None


def refresh_scenario_leaderboard(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT asset_kind, horizon_hours, strategy_family, is_champion,
               skill_score, nrmse, pinball
        FROM scenario_runs
        """
    ).fetchall()
    grouped: dict[tuple[str, int, str], list[tuple]] = {}
    for row in rows:
        key = (row[0], int(row[1]), row[2])
        grouped.setdefault(key, []).append(row)

    con.execute("DELETE FROM scenario_leaderboard")
    for (asset_kind, horizon_hours, family), items in grouped.items():
        trials = len(items)
        wins = sum(1 for item in items if bool(item[3]))
        con.execute(
            """
            INSERT INTO scenario_leaderboard
              (asset_kind, horizon_hours, strategy_family, trials, wins, win_rate,
               median_skill_score, median_nrmse, median_pinball)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                asset_kind,
                horizon_hours,
                family,
                trials,
                wins,
                wins / trials if trials else 0.0,
                _median([item[4] for item in items]),
                _median([item[5] for item in items]),
                _median([item[6] for item in items]),
            ],
        )


def scenario_leaderboard(
    con: duckdb.DuckDBPyConnection,
    *,
    asset_kind: str | None = None,
    horizon_hours: int | None = None,
) -> list[dict]:
    query = (
        "SELECT asset_kind, horizon_hours, strategy_family, trials, wins, win_rate, "
        "median_skill_score, median_nrmse, median_pinball, last_updated_at "
        "FROM scenario_leaderboard WHERE 1=1"
    )
    params: list[object] = []
    if asset_kind is not None:
        query += " AND asset_kind=?"
        params.append(asset_kind)
    if horizon_hours is not None:
        query += " AND horizon_hours=?"
        params.append(horizon_hours)
    query += " ORDER BY win_rate DESC, median_nrmse ASC NULLS LAST, trials DESC"
    result = con.execute(query, params)
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def policy_leaderboard(
    con: duckdb.DuckDBPyConnection,
    *,
    asset_kind: str | None = None,
    horizon_hours: int | None = None,
    plant_id: str | None = None,
) -> list[dict]:
    """Aggregate wins / median skill by {point_policy × source_policy} combo.

    Answers the roadmap's "which spatial/source combination wins" question at a
    coarser grain than :func:`scenario_leaderboard` (which groups by full
    strategy_family). Ordered best-first: highest win_rate, then highest median
    skill, then lowest median nRMSE.
    """
    query = (
        "SELECT coalesce(point_policy, 'single_point') AS point_policy, "
        "coalesce(source_policy, 'single_source') AS source_policy, "
        "count(*) AS trials, "
        "sum(CASE WHEN is_champion THEN 1 ELSE 0 END) AS wins, "
        "sum(CASE WHEN is_champion THEN 1 ELSE 0 END) * 1.0 / count(*) AS win_rate, "
        "median(skill_score) AS median_skill_score, "
        "median(nrmse) AS median_nrmse, "
        "median(pinball) AS median_pinball "
        "FROM scenario_runs WHERE 1=1"
    )
    params: list[object] = []
    if asset_kind is not None:
        query += " AND asset_kind=?"
        params.append(asset_kind)
    if horizon_hours is not None:
        query += " AND horizon_hours=?"
        params.append(horizon_hours)
    if plant_id is not None:
        query += " AND plant_id=?"
        params.append(plant_id)
    query += (
        " GROUP BY 1, 2 "
        "ORDER BY win_rate DESC, median_skill_score DESC NULLS LAST, "
        "median_nrmse ASC NULLS LAST, trials DESC, point_policy, source_policy"
    )
    result = con.execute(query, params)
    cols = [d[0] for d in result.description]
    rows = []
    for row in result.fetchall():
        record = dict(zip(cols, row))
        record["trials"] = int(record["trials"])
        record["wins"] = int(record["wins"])
        record["win_rate"] = float(record["win_rate"])
        rows.append(record)
    return rows


def target_leaderboard(
    con: duckdb.DuckDBPyConnection,
    *,
    asset_kind: str | None = None,
    horizon_hours: int | None = None,
    plant_id: str | None = None,
) -> list[dict]:
    """Aggregate wins / median skill / median peak_mae by target_policy (C1-t7).

    Answers "does kpv beat capacity_norm on peak_mae/skill for this plant?" —
    the acceptance question for the kPV target transform. Ordered best-first:
    highest win_rate, then highest median skill, then lowest median peak_mae.
    """
    query = (
        "SELECT coalesce(target_policy, 'capacity_norm') AS target_policy, "
        "count(*) AS trials, "
        "sum(CASE WHEN is_champion THEN 1 ELSE 0 END) AS wins, "
        "sum(CASE WHEN is_champion THEN 1 ELSE 0 END) * 1.0 / count(*) AS win_rate, "
        "median(skill_score) AS median_skill_score, "
        "median(peak_mae) AS median_peak_mae, "
        "median(nrmse) AS median_nrmse "
        "FROM scenario_runs WHERE 1=1"
    )
    params: list[object] = []
    if asset_kind is not None:
        query += " AND asset_kind=?"
        params.append(asset_kind)
    if horizon_hours is not None:
        query += " AND horizon_hours=?"
        params.append(horizon_hours)
    if plant_id is not None:
        query += " AND plant_id=?"
        params.append(plant_id)
    query += (
        " GROUP BY 1 "
        "ORDER BY win_rate DESC, median_skill_score DESC NULLS LAST, "
        "median_peak_mae ASC NULLS LAST, trials DESC, target_policy"
    )
    result = con.execute(query, params)
    cols = [d[0] for d in result.description]
    rows = []
    for row in result.fetchall():
        record = dict(zip(cols, row))
        record["trials"] = int(record["trials"])
        record["wins"] = int(record["wins"])
        record["win_rate"] = float(record["win_rate"])
        rows.append(record)
    return rows


def plant_strategy_profile(con: duckdb.DuckDBPyConnection, plant_id: str) -> list[dict]:
    result = con.execute(
        """
        SELECT horizon_hours, strategy_family, count(*) AS trials,
               sum(CASE WHEN is_champion THEN 1 ELSE 0 END) AS wins,
               median(nrmse) AS median_nrmse,
               median(skill_score) AS median_skill_score,
               max(created_at) AS updated_at
        FROM scenario_runs
        WHERE plant_id=?
        GROUP BY horizon_hours, strategy_family
        ORDER BY horizon_hours, wins DESC, median_nrmse ASC NULLS LAST
        """,
        [plant_id],
    )
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]
