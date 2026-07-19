from __future__ import annotations

import json

import duckdb

from openenergy.experiments.search_space import PipelineConfig
from openenergy.experiments.strategy import StrategyConfig


def create_experiment(con: duckdb.DuckDBPyConnection, plant_id: str, horizons: list[int],
                      objective: str = "nrmse") -> int:
    return con.execute(
        "INSERT INTO experiments (plant_id, horizons, objective, status) VALUES (?,?,?,'running') RETURNING experiment_id",
        [plant_id, ",".join(map(str, horizons)), objective],
    ).fetchone()[0]


def record_trial(con: duckdb.DuckDBPyConnection, *, experiment_id: int, horizon_hours: int,
                 pipeline_config: PipelineConfig, cv_nrmse: float | None, test_metrics: dict,
                 skill_score: float, clean_metrics: dict | None = None,
                 clean_skill_score: float | None = None,
                 quality_audit: dict | None = None) -> int:
    clean_metrics = clean_metrics or {}
    quality_audit = quality_audit or {}
    return con.execute(
        """
        INSERT INTO experiment_trials
          (experiment_id, horizon_hours, model_family, nwp_source, feature_blocks, params,
           cv_nrmse, test_nrmse, test_nmae, test_bias, test_pinball,
           test_coverage, test_interval_width, test_crps, test_peak_bias, test_peak_mae,
           skill_score, test_clean_nrmse, clean_skill_score,
           train_all_count, train_clean_count, eval_all_count, eval_clean_count,
           is_champion)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,false) RETURNING trial_id
        """,
        [experiment_id, horizon_hours, pipeline_config.model_family, pipeline_config.nwp_source,
         json.dumps(pipeline_config.feature_blocks), json.dumps(pipeline_config.params),
         cv_nrmse, test_metrics.get("nrmse"), test_metrics.get("nmae"), test_metrics.get("bias"),
         test_metrics.get("pinball"), test_metrics.get("coverage"), test_metrics.get("interval_width"),
         test_metrics.get("crps"), test_metrics.get("peak_bias"), test_metrics.get("peak_mae"),
         skill_score, clean_metrics.get("nrmse"), clean_skill_score,
         quality_audit.get("train_all_count"), quality_audit.get("train_clean_count"),
         quality_audit.get("eval_all_count"), quality_audit.get("eval_clean_count")],
    ).fetchone()[0]


def record_strategy_trial(
    con: duckdb.DuckDBPyConnection,
    *,
    experiment_id: int,
    horizon_hours: int,
    strategy_config: StrategyConfig,
    cv_nrmse: float | None,
    test_metrics: dict,
    skill_score: float,
    artifact_path: str | None,
    selection_diagnostics: dict | None = None,
    uncertainty_diagnostics: dict | None = None,
    evaluation_grade: str = "exploratory",
    evaluation_n_days: int | None = None,
    evaluation_origins: int | None = None,
    evaluation_seasons: int | None = None,
    clean_metrics: dict | None = None,
    clean_skill_score: float | None = None,
    quality_audit: dict | None = None,
) -> int:
    selection_diagnostics = selection_diagnostics or {}
    uncertainty_diagnostics = uncertainty_diagnostics or {}
    clean_metrics = clean_metrics or {}
    quality_audit = quality_audit or {}
    return con.execute(
        """
        INSERT INTO strategy_trials
          (experiment_id, horizon_hours, ensemble_method, strategy_config,
           cv_nrmse, test_nrmse, test_nmae, test_bias, test_pinball,
           test_coverage, test_interval_width, test_crps, test_peak_bias, test_peak_mae,
           band_status, sel_coverage, sel_coverage_lower, sel_coverage_upper,
           sel_winkler, sel_kupiec_pvalue, sel_independence_pvalue,
           sel_conditional_coverage_pvalue, sel_n,
           champion_metric_name, champion_metric_lower, champion_metric_upper,
           mcs_members, uncertainty_diagnostics,
           evaluation_grade, evaluation_n_days, evaluation_origins, evaluation_seasons,
           skill_score, test_clean_nrmse, clean_skill_score,
           train_all_count, train_clean_count, eval_all_count, eval_clean_count,
           artifact_path, is_champion)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,false)
        RETURNING strategy_trial_id
        """,
        [
            experiment_id,
            horizon_hours,
            strategy_config.ensemble_method,
            json.dumps({
                "ensemble_method": strategy_config.ensemble_method,
                "top_k": strategy_config.top_k,
                "candidates": [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in strategy_config.candidates
                ],
            }),
            cv_nrmse,
            test_metrics.get("nrmse"),
            test_metrics.get("nmae"),
            test_metrics.get("bias"),
            test_metrics.get("pinball"),
            test_metrics.get("coverage"),
            test_metrics.get("interval_width"),
            test_metrics.get("crps"),
            test_metrics.get("peak_bias"),
            test_metrics.get("peak_mae"),
            selection_diagnostics.get("band_status", "unknown"),
            selection_diagnostics.get("sel_coverage"),
            selection_diagnostics.get("sel_coverage_lower"),
            selection_diagnostics.get("sel_coverage_upper"),
            selection_diagnostics.get("sel_winkler"),
            selection_diagnostics.get("sel_kupiec_pvalue"),
            selection_diagnostics.get("sel_independence_pvalue"),
            selection_diagnostics.get("sel_conditional_coverage_pvalue"),
            selection_diagnostics.get("sel_n"),
            uncertainty_diagnostics.get("champion_metric"),
            uncertainty_diagnostics.get("champion_metric_lower"),
            uncertainty_diagnostics.get("champion_metric_upper"),
            json.dumps(uncertainty_diagnostics.get("mcs_members", [])),
            json.dumps(uncertainty_diagnostics, sort_keys=True),
            evaluation_grade,
            evaluation_n_days,
            evaluation_origins,
            evaluation_seasons,
            skill_score,
            clean_metrics.get("nrmse"),
            clean_skill_score,
            quality_audit.get("train_all_count"),
            quality_audit.get("train_clean_count"),
            quality_audit.get("eval_all_count"),
            quality_audit.get("eval_clean_count"),
            artifact_path,
        ],
    ).fetchone()[0]


def record_strategy_candidate(
    con: duckdb.DuckDBPyConnection,
    *,
    strategy_trial_id: int,
    rank: int,
    candidate_config: PipelineConfig,
    cv_nrmse: float | None,
    test_nrmse: float | None,
    artifact_path: str | None,
    feature_names: list[str],
) -> int:
    return con.execute(
        """
        INSERT INTO strategy_candidates
          (strategy_trial_id, rank, candidate_config, cv_nrmse,
           test_nrmse, artifact_path, feature_names)
        VALUES (?,?,?,?,?,?,?) RETURNING strategy_candidate_id
        """,
        [
            strategy_trial_id,
            rank,
            json.dumps(candidate_config.model_dump()),
            cv_nrmse,
            test_nrmse,
            artifact_path,
            json.dumps(feature_names),
        ],
    ).fetchone()[0]


def mark_champion(con: duckdb.DuckDBPyConnection, experiment_id: int, horizon_hours: int, trial_id: int) -> None:
    con.execute(
        "UPDATE experiment_trials SET is_champion=false WHERE experiment_id=? AND horizon_hours=?",
        [experiment_id, horizon_hours])
    con.execute("UPDATE experiment_trials SET is_champion=true WHERE trial_id=?", [trial_id])


def mark_strategy_champion(
    con: duckdb.DuckDBPyConnection,
    experiment_id: int,
    horizon_hours: int,
    strategy_trial_id: int,
) -> None:
    con.execute(
        "UPDATE strategy_trials SET is_champion=false WHERE experiment_id=? AND horizon_hours=?",
        [experiment_id, horizon_hours],
    )
    con.execute(
        "UPDATE strategy_trials SET is_champion=true WHERE strategy_trial_id=?",
        [strategy_trial_id],
    )


def promote_strategy_champion(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int, strategy_trial_id: int
) -> None:
    """Point the served strategy champion for (plant, horizon) at ``strategy_trial_id``.

    The single source of truth for the forecast path (T-05). Upserted so exactly
    one current row exists per (plant, horizon); an explicit promotion (bootstrap
    or a DM-gated retrain win) is the only thing that moves it."""
    con.execute(
        "INSERT OR REPLACE INTO champion_pointer "
        "(plant_id, horizon_hours, strategy_trial_id) VALUES (?,?,?)",
        [plant_id, horizon_hours, strategy_trial_id],
    )


def get_pointer_strategy_trial(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int
) -> int | None:
    """Return the currently-served strategy_trial_id from champion_pointer, or None."""
    row = con.execute(
        "SELECT strategy_trial_id FROM champion_pointer WHERE plant_id=? AND horizon_hours=?",
        [plant_id, horizon_hours],
    ).fetchone()
    return row[0] if row is not None else None


def get_champion_strategy_model(
    con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int
) -> dict | None:
    # Serve strictly what champion_pointer names (T-05) — NOT strategy_trials.is_champion,
    # which run_experiment sets optimistically per run and would serve a rejected challenger.
    result = con.execute(
        """
        SELECT st.strategy_trial_id, st.artifact_path, st.strategy_config, st.experiment_id,
               st.evaluation_grade, st.evaluation_n_days, st.evaluation_origins,
               st.evaluation_seasons
        FROM champion_pointer cp
        JOIN strategy_trials st ON st.strategy_trial_id = cp.strategy_trial_id
        WHERE cp.plant_id = ? AND cp.horizon_hours = ?
        LIMIT 1
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    if result is None:
        return None
    (
        strategy_trial_id,
        artifact_path,
        strategy_config,
        experiment_id,
        evaluation_grade_value,
        evaluation_n_days,
        evaluation_origins,
        evaluation_seasons,
    ) = result
    return {
        "strategy_trial_id": strategy_trial_id,
        "artifact_path": artifact_path,
        "strategy_config": json.loads(strategy_config),
        "experiment_id": experiment_id,
        "evaluation_grade": evaluation_grade_value,
        "evaluation_n_days": evaluation_n_days,
        "evaluation_origins": evaluation_origins,
        "evaluation_seasons": evaluation_seasons,
    }


def get_champion_model(con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int) -> dict | None:
    """Return the model belonging to the strategy named by ``champion_pointer``.

    The legacy model row remains useful as a serving fallback for old strategy
    artifacts, but it must be resolved *through* the served strategy pointer.
    Looking up ``models.is_champion`` directly would create a second serving
    source of truth and could expose an optimistically-marked challenger.
    """
    result = con.execute(
        """
        SELECT m.model_id, m.artifact_path, m.feature_names, m.metrics, m.experiment_id
        FROM champion_pointer cp
        JOIN strategy_trials st
          ON st.strategy_trial_id = cp.strategy_trial_id
         AND st.horizon_hours = cp.horizon_hours
        JOIN models m
          ON m.experiment_id = st.experiment_id
         AND m.horizon_hours = cp.horizon_hours
        WHERE cp.plant_id = ? AND cp.horizon_hours = ? AND m.is_champion = true
        ORDER BY m.model_id DESC
        LIMIT 1
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    if result is None:
        return None
    model_id, artifact_path, feature_names_json, metrics_json, experiment_id = result
    return {
        "model_id": model_id,
        "artifact_path": artifact_path,
        "feature_names": json.loads(feature_names_json) if feature_names_json else [],
        "metrics": json.loads(metrics_json) if metrics_json else {},
        "experiment_id": experiment_id,
    }


def get_champion_feature_blocks(con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int) -> list[str]:
    """Return feature blocks for the experiment named by ``champion_pointer``.

    This supports the legacy model fallback without consulting a global
    ``experiment_trials.is_champion`` flag as an independent serving selector.
    Raises ValueError if the pointer has no matching champion trial.
    """
    result = con.execute(
        """
        SELECT et.feature_blocks
        FROM champion_pointer cp
        JOIN strategy_trials st
          ON st.strategy_trial_id = cp.strategy_trial_id
         AND st.horizon_hours = cp.horizon_hours
        JOIN experiment_trials et
          ON et.experiment_id = st.experiment_id
         AND et.horizon_hours = cp.horizon_hours
        WHERE cp.plant_id = ? AND cp.horizon_hours = ? AND et.is_champion = true
        ORDER BY et.trial_id DESC
        LIMIT 1
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    if result is None:
        raise ValueError(
            f"No champion trial found for plant_id={plant_id!r}, horizon_hours={horizon_hours}"
        )
    return json.loads(result[0]) if result[0] else []


def best_trials(con: duckdb.DuckDBPyConnection, experiment_id: int) -> list[dict]:
    strategy_rows = con.execute(
        """
        SELECT horizon_hours, strategy_trial_id AS trial_id,
               'strategy:' || ensemble_method AS model_family,
               'strategy' AS nwp_source,
               strategy_config AS feature_blocks,
               strategy_config AS params,
               cv_nrmse, test_nrmse, test_nmae, test_bias, test_pinball,
               test_coverage, test_interval_width, test_crps, test_peak_bias, test_peak_mae,
               band_status, sel_coverage, sel_coverage_lower, sel_coverage_upper,
               sel_winkler, sel_kupiec_pvalue, sel_independence_pvalue,
               sel_conditional_coverage_pvalue, sel_n,
               evaluation_grade, evaluation_n_days, evaluation_origins,
               evaluation_seasons,
               skill_score, test_clean_nrmse, clean_skill_score,
               train_all_count, train_clean_count, eval_all_count, eval_clean_count,
               is_champion
        FROM strategy_trials WHERE experiment_id=?
        QUALIFY row_number() OVER (PARTITION BY horizon_hours ORDER BY test_nrmse ASC NULLS LAST) = 1
        """,
        [experiment_id],
    )
    strategy_cols = [d[0] for d in strategy_rows.description]
    strategy = [dict(zip(strategy_cols, row)) for row in strategy_rows.fetchall()]
    if strategy:
        return strategy

    result = con.execute(
        """
        SELECT horizon_hours, trial_id, model_family, nwp_source, feature_blocks, params,
               cv_nrmse, test_nrmse, test_nmae, test_bias, test_pinball,
               test_coverage, test_interval_width, test_crps, test_peak_bias, test_peak_mae,
               NULL AS band_status, NULL AS sel_coverage,
               NULL AS sel_coverage_lower, NULL AS sel_coverage_upper,
               NULL AS sel_winkler, NULL AS sel_kupiec_pvalue,
               NULL AS sel_independence_pvalue,
               NULL AS sel_conditional_coverage_pvalue, NULL AS sel_n,
               'exploratory' AS evaluation_grade,
               NULL AS evaluation_n_days, NULL AS evaluation_origins,
               NULL AS evaluation_seasons,
               skill_score, test_clean_nrmse, clean_skill_score,
               train_all_count, train_clean_count, eval_all_count, eval_clean_count,
               is_champion
        FROM experiment_trials WHERE experiment_id=?
        QUALIFY row_number() OVER (PARTITION BY horizon_hours ORDER BY test_nrmse ASC NULLS LAST) = 1
        """,
        [experiment_id])
    cols = [d[0] for d in result.description]
    rows = result.fetchall()
    return [dict(zip(cols, row)) for row in rows]
