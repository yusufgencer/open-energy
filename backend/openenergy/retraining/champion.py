from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from openenergy.evaluation.metrics import dm_test
from openenergy.evaluation.splits import rolling_origin_splits, walk_forward_split
from openenergy.experiments.assembly import assemble_raw
from openenergy.experiments.persistence import promote_strategy_champion
from openenergy.experiments.runner import _fit_candidate, _norm_prediction, run_experiment
from openenergy.experiments.scenario_memory import (
    refresh_scenario_leaderboard,
    update_scenario_champion,
)
from openenergy.experiments.search_space import PipelineConfig

# Significance level for the Diebold-Mariano champion-challenger gate. A challenger
# must beat the incumbent on scalar nRMSE AND clear this two-sided DM threshold on
# per-timestamp squared-error differentials before it may dethrone the champion.
DM_ALPHA = 0.05


def significance_gate(
    y_true: np.ndarray,
    incumbent_p50: np.ndarray,
    challenger_p50: np.ndarray,
    *,
    horizon: int,
    alpha: float = DM_ALPHA,
) -> tuple[bool, float, float]:
    """Diebold-Mariano gate on paired per-timestamp point forecasts.

    Loss differential ``d[t] = (y-incumbent)^2 - (y-challenger)^2`` — positive when
    the challenger is closer. Returns ``(passed, p_value, dm_stat)`` where
    ``passed`` is True only when the challenger is better in the mean AND the DM
    test rejects equal predictive accuracy at ``alpha`` (two-sided).
    """
    y_true = np.asarray(y_true, dtype=float)
    incumbent_p50 = np.asarray(incumbent_p50, dtype=float)
    challenger_p50 = np.asarray(challenger_p50, dtype=float)
    loss_diff = (y_true - incumbent_p50) ** 2 - (y_true - challenger_p50) ** 2
    dm_stat, p_value = dm_test(loss_diff, h=horizon)
    passed = bool(float(np.mean(loss_diff)) > 0.0 and p_value < alpha)
    return passed, p_value, dm_stat


def _config_from_trial(con: duckdb.DuckDBPyConnection, trial_id: int) -> PipelineConfig | None:
    """Rebuild a representative single-model PipelineConfig from a stored trial.

    experiment_trials persists the champion's best single candidate (model_family,
    nwp_source, feature_blocks, params); policy axes default to the legacy
    capacity_norm path. Enough to refit a comparable point model for the DM gate.
    """
    row = con.execute(
        "SELECT model_family, nwp_source, feature_blocks, params "
        "FROM experiment_trials WHERE trial_id = ?",
        [trial_id],
    ).fetchone()
    if row is None:
        return None
    model_family, nwp_source, feature_blocks_json, params_json = row
    if model_family is None:
        return None
    return PipelineConfig(
        model_family=model_family,
        nwp_source=nwp_source or "best_match",
        feature_blocks=json.loads(feature_blocks_json) if feature_blocks_json else [],
        params=json.loads(params_json) if params_json else {},
    )


def _refit_p50_on_test(raw, cfg: PipelineConfig, *, kind: str, capacity_mw: float,
                       tr_idx: np.ndarray, te_idx: np.ndarray) -> np.ndarray:
    """Fit ``cfg`` on the train split and return normalized-power p50 on the test split."""
    ds, model, _y_norm, p_cs = _fit_candidate(
        raw, cfg, kind=kind, capacity_mw=capacity_mw, train_idx=tr_idx,
    )
    pred = model.predict(ds.X[te_idx])
    pred = _norm_prediction(pred, cfg, p_cs[te_idx] if p_cs is not None else None, capacity_mw)
    return np.asarray(pred.p50, dtype=float)


def _dm_gate_horizon(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                     horizon: int, capacity_mw: float, kind: str,
                     old_trial_id: int, new_trial_id: int,
                     embargo: int = 24) -> tuple[bool, float | None]:
    """Reconstruct paired incumbent/challenger point forecasts on a shared held-out
    test window and run the significance gate.

    Returns ``(significant, p_value)``. On any reconstruction failure (too little
    data, unrecoverable config) it degrades to ``(True, None)`` so the scalar gate
    alone governs — DM only ever *blocks* a scalar win, never fabricates one.
    """
    try:
        old_cfg = _config_from_trial(con, old_trial_id)
        new_cfg = _config_from_trial(con, new_trial_id)
        if old_cfg is None or new_cfg is None:
            return True, None
        raw = assemble_raw(con, plant_id=plant_id, point_id=point_id, horizon_hours=horizon)
        n = raw.height
        if n < 30:
            return True, None
        # T-22: align incumbent and challenger losses over the same fixed-window
        # rolling origins used for experiment reporting.  Very small histories
        # retain the legacy tail holdout rather than turning lack of data into a
        # fabricated significance result.
        try:
            folds = rolling_origin_splits(
                raw["valid_time"].to_numpy(),
                n_splits=3,
                test_size=max(2, n // 10),
                embargo=embargo,
            )
            if len(folds[0][0]) < 10:
                raise ValueError("rolling-origin train window too short")
        except ValueError:
            ts = max(1, n // 5)
            if ts < 2:
                return True, None
            folds = [walk_forward_split(n, test_size=ts)]

        truths: list[np.ndarray] = []
        incumbent_predictions: list[np.ndarray] = []
        challenger_predictions: list[np.ndarray] = []
        for tr_idx, te_idx in folds:
            truths.append(np.asarray(
                _fit_candidate(
                    raw, new_cfg, kind=kind, capacity_mw=capacity_mw,
                    train_idx=tr_idx,
                )[2],
                dtype=float,
            )[te_idx])
            incumbent_predictions.append(_refit_p50_on_test(
                raw, old_cfg, kind=kind, capacity_mw=capacity_mw,
                tr_idx=tr_idx, te_idx=te_idx,
            ))
            challenger_predictions.append(_refit_p50_on_test(
                raw, new_cfg, kind=kind, capacity_mw=capacity_mw,
                tr_idx=tr_idx, te_idx=te_idx,
            ))
        significant, p_value, _stat = significance_gate(
            np.concatenate(truths),
            np.concatenate(incumbent_predictions),
            np.concatenate(challenger_predictions),
            horizon=horizon,
        )
        return significant, p_value
    except Exception:
        return True, None


def _get_incumbent_full(con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int) -> dict:
    """Return {nrmse, trial_id, model_id} of current champion (all may be None if no champion)."""
    trial_row = con.execute(
        """
        SELECT et.trial_id, et.test_nrmse
        FROM experiment_trials et
        JOIN experiments e ON et.experiment_id = e.experiment_id
        WHERE e.plant_id = ? AND et.horizon_hours = ? AND et.is_champion = true
        ORDER BY et.trial_id DESC LIMIT 1
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    model_row = con.execute(
        """
        SELECT m.model_id FROM models m
        JOIN experiments e ON m.experiment_id = e.experiment_id
        WHERE e.plant_id = ? AND m.horizon_hours = ? AND m.is_champion = true
        ORDER BY m.model_id DESC LIMIT 1
        """,
        [plant_id, horizon_hours],
    ).fetchone()
    return {
        "trial_id": trial_row[0] if trial_row else None,
        "nrmse": trial_row[1] if trial_row else None,
        "model_id": model_row[0] if model_row else None,
    }


def retrain(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
            horizon_hours_list: list[int], capacity_mw: float, kind: str,
            nwp_sources: list[str], n_trials: int, models_dir: str | Path,
            n_splits: int = 4, embargo: int = 24, seed: int = 42) -> dict[int, dict]:
    """Retrain and promote champion only if new test_nrmse is strictly better.

    Returns dict mapping horizon_hours -> {promoted: bool, new_nrmse: float|None, old_nrmse: float|None}

    Logic:
    1. Read incumbent champion test_nrmse, trial_id, and model_id per horizon.
    2. Run run_experiment (new experiment; its mark_champion step globally demotes old champions).
    3. For each horizon:
       - If no incumbent (first run): always promote (new champion stays).
       - If new_nrmse < incumbent_nrmse (strictly): promote (new champion stays).
       - Else: demote new champion trial+model, explicitly restore old champion trial+model.
    """
    models_dir = Path(models_dir)

    # Read incumbent info BEFORE running new experiment (run_experiment globally demotes old ones)
    incumbents: dict[int, dict] = {
        h: _get_incumbent_full(con, plant_id, h) for h in horizon_hours_list
    }

    # Run new experiment (produces new champion rows per horizon)
    new_eid = run_experiment(
        con, plant_id=plant_id, point_id=point_id,
        horizon_hours_list=horizon_hours_list, capacity_mw=capacity_mw,
        kind=kind, nwp_sources=nwp_sources, n_trials=n_trials,
        n_splits=n_splits, embargo=embargo, seed=seed, models_dir=models_dir,
    )

    outcome: dict[int, dict] = {}

    for horizon in horizon_hours_list:
        inc = incumbents[horizon]
        old_nrmse = inc["nrmse"]
        old_trial_id = inc["trial_id"]
        old_model_id = inc["model_id"]

        # Get new champion trial for this horizon from the new experiment
        new_champ_row = con.execute(
            """
            SELECT trial_id, test_nrmse
            FROM experiment_trials
            WHERE experiment_id = ? AND horizon_hours = ? AND is_champion = true
            """,
            [new_eid, horizon],
        ).fetchone()

        if new_champ_row is None:
            # No champion was produced (insufficient data, sentinel CV)
            # Restore old incumbent if run_experiment demoted it globally
            if old_trial_id is not None:
                con.execute(
                    "UPDATE experiment_trials SET is_champion=true WHERE trial_id=?",
                    [old_trial_id],
                )
            if old_model_id is not None:
                con.execute("UPDATE models SET is_champion=true WHERE model_id=?", [old_model_id])
            outcome[horizon] = {"promoted": False, "new_nrmse": None,
                                "old_nrmse": old_nrmse, "p_value": None}
            continue

        new_trial_id, new_nrmse = new_champ_row

        # Incumbent is stale/broken when it has no recorded skill (first run or a
        # NaN nRMSE) — promote unconditionally, skipping the significance gate.
        incumbent_broken = old_nrmse is None or (
            isinstance(old_nrmse, float) and np.isnan(old_nrmse)
        )
        scalar_better = new_nrmse is not None and (
            incumbent_broken or new_nrmse < old_nrmse
        )

        # A genuine scalar win over a live incumbent must ALSO be statistically
        # significant (Diebold-Mariano on paired per-timestamp errors) — noise
        # alone must not dethrone a proven champion (E1-t1).
        p_value: float | None = None
        significant = True
        if scalar_better and not incumbent_broken and old_trial_id is not None:
            significant, p_value = _dm_gate_horizon(
                con, plant_id=plant_id, point_id=point_id, horizon=horizon,
                capacity_mw=capacity_mw, kind=kind,
                old_trial_id=old_trial_id, new_trial_id=new_trial_id,
                embargo=embargo,
            )

        if scalar_better and significant:
            # run_experiment already globally demoted old champions for this plant+horizon;
            # ensure any remaining are demoted (idempotent safety net).
            con.execute(
                """
                UPDATE experiment_trials SET is_champion=false
                WHERE horizon_hours = ? AND is_champion = true AND trial_id != ?
                AND experiment_id IN (
                    SELECT experiment_id FROM experiments WHERE plant_id = ?
                )
                """,
                [horizon, new_trial_id, plant_id],
            )
            con.execute(
                """
                UPDATE models SET is_champion=false
                WHERE horizon_hours = ? AND is_champion = true
                AND experiment_id IN (
                    SELECT experiment_id FROM experiments WHERE plant_id = ?
                )
                AND experiment_id != ?
                """,
                [horizon, plant_id, new_eid],
            )
            # Move the served strategy pointer to the promoted run's strategy
            # champion — the DM gate now governs the forecast path, not just the
            # offline model tables (T-05).
            new_strategy_row = con.execute(
                "SELECT strategy_trial_id FROM strategy_trials "
                "WHERE experiment_id=? AND horizon_hours=? AND is_champion=true "
                "ORDER BY strategy_trial_id DESC LIMIT 1",
                [new_eid, horizon],
            ).fetchone()
            if new_strategy_row is not None:
                promote_strategy_champion(con, plant_id, horizon, new_strategy_row[0])
            outcome[horizon] = {"promoted": True, "new_nrmse": float(new_nrmse),
                                "old_nrmse": old_nrmse, "p_value": p_value}
        else:
            # Challenger does NOT beat incumbent; demote new champion trial and model
            con.execute(
                "UPDATE experiment_trials SET is_champion=false WHERE trial_id=?",
                [new_trial_id],
            )
            con.execute(
                "UPDATE models SET is_champion=false WHERE experiment_id=? AND horizon_hours=?",
                [new_eid, horizon],
            )
            # Restore old champion explicitly (run_experiment may have globally demoted it)
            if old_trial_id is not None:
                con.execute(
                    "UPDATE experiment_trials SET is_champion=true WHERE trial_id=?",
                    [old_trial_id],
                )
            if old_model_id is not None:
                con.execute("UPDATE models SET is_champion=true WHERE model_id=?", [old_model_id])
            outcome[horizon] = {
                "promoted": False,
                "new_nrmse": float(new_nrmse) if new_nrmse is not None else None,
                "old_nrmse": old_nrmse,
                "p_value": p_value,
            }

    # E1-t5: reconcile each horizon's scenario_runs row (recorded optimistically as
    # is_champion=True by run_experiment) with the actual promote/defend decision so
    # the leaderboard accumulates operational evidence, not offline optimism.
    for horizon in horizon_hours_list:
        update_scenario_champion(
            con,
            experiment_id=new_eid,
            horizon_hours=horizon,
            is_champion=bool(outcome[horizon]["promoted"]),
        )
    refresh_scenario_leaderboard(con)

    return outcome
