"""Serving-only dynamic adaptive conformal inference (T-21).

The offline search remains blind to this layer.  At serving time, each
``(plant_id, horizon_hours)`` stream first consumes newly realised historical
forecast/actual pairs in ``valid_time`` order, then adjusts the new band.  A
feedback ledger makes replaying a valid time idempotent and the persisted state
plus issued-band lineage makes every update auditable.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

import duckdb
import numpy as np

from openenergy.models.base import Prediction, enforce_quantile_order

DEFAULT_ALPHA = 0.20
DEFAULT_GAMMA = 0.02
DEFAULT_SCORE_WINDOW = 256
DEFAULT_MIN_HISTORY = 8


@dataclass
class OnlineDtACI:
    """Small DtACI-like state machine using only one-step-delayed feedback.

    ``alpha_t`` follows the ACI coverage-error update while the width correction
    is a rolling empirical score quantile.  The rolling window lets the score
    distribution itself track shifts; clipping ``alpha_t`` avoids infinite
    serving bands during startup or a burst of misses.
    """

    target_alpha: float = DEFAULT_ALPHA
    alpha_t: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    score_window: int = DEFAULT_SCORE_WINDOW
    min_history: int = DEFAULT_MIN_HISTORY
    scores: list[float] = field(default_factory=list)
    feedback_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_alpha < 1.0:
            raise ValueError("target_alpha must be in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.score_window < 1:
            raise ValueError("score_window must be positive")

    def adjustment(self) -> float:
        if len(self.scores) < self.min_history:
            return 0.0
        level = 1.0 - float(np.clip(self.alpha_t, 1e-3, 1.0 - 1e-3))
        return float(np.quantile(np.asarray(self.scores), level, method="higher"))

    def adjust(self, pred: Prediction) -> Prediction:
        if pred.p10 is None or pred.p90 is None:
            return pred
        q = self.adjustment()
        lo = np.asarray(pred.p10, dtype=float) - q
        hi = np.asarray(pred.p90, dtype=float) + q
        return enforce_quantile_order(
            Prediction(
                p50=np.asarray(pred.p50, dtype=float),
                p10=np.minimum(lo, hi),
                p90=np.maximum(lo, hi),
            )
        )

    def update(
        self,
        *,
        actual: float,
        base_lower: float,
        base_upper: float,
        served_lower: float,
        served_upper: float,
    ) -> tuple[float, float]:
        """Consume one realised pair and return ``(score, miss_indicator)``."""
        error = float(not (served_lower <= actual <= served_upper))
        score = float(max(base_lower - actual, actual - base_upper))
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.target_alpha - error),
                1e-3,
                1.0 - 1e-3,
            )
        )
        self.scores.append(score)
        if len(self.scores) > self.score_window:
            self.scores = self.scores[-self.score_window :]
        self.feedback_count += 1
        return score, error


def _load_state(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    horizon_hours: int,
) -> tuple[OnlineDtACI, dt.datetime | None, bool]:
    row = con.execute(
        """SELECT target_alpha, alpha_t, gamma, score_window, scores,
                  feedback_count, last_feedback_valid_time
           FROM online_conformal_state
           WHERE plant_id=? AND horizon_hours=?""",
        [plant_id, horizon_hours],
    ).fetchone()
    if row is None:
        return OnlineDtACI(), None, False
    return (
        OnlineDtACI(
            target_alpha=float(row[0]),
            alpha_t=float(row[1]),
            gamma=float(row[2]),
            score_window=int(row[3]),
            scores=[float(v) for v in json.loads(row[4])],
            feedback_count=int(row[5]),
        ),
        row[6],
        True,
    )


def _save_state(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    state: OnlineDtACI,
    last_feedback_valid_time: dt.datetime | None,
) -> None:
    con.execute(
        """INSERT INTO online_conformal_state
           (plant_id, horizon_hours, target_alpha, alpha_t, gamma, score_window,
            scores, feedback_count, last_feedback_valid_time)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT (plant_id, horizon_hours) DO UPDATE SET
             target_alpha=excluded.target_alpha, alpha_t=excluded.alpha_t,
             gamma=excluded.gamma, score_window=excluded.score_window,
             scores=excluded.scores, feedback_count=excluded.feedback_count,
             last_feedback_valid_time=excluded.last_feedback_valid_time,
             updated_at=now()""",
        [
            plant_id,
            horizon_hours,
            state.target_alpha,
            state.alpha_t,
            state.gamma,
            state.score_window,
            json.dumps(state.scores),
            state.feedback_count,
            last_feedback_valid_time,
        ],
    )


def refresh_and_apply(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    issue_time: dt.datetime,
    pred: Prediction,
) -> tuple[Prediction, int]:
    """Consume unseen past feedback, persist state, and adjust ``pred``.

    The anti-join against ``online_conformal_feedback`` is the idempotency
    boundary.  ``f.valid_time < issue_time`` is the causality boundary. Existing
    forecasts from before T-21 remain eligible: absent issued lineage, their
    stored band is treated as both base and served.
    """
    state, last_valid_time, state_exists = _load_state(con, plant_id, horizon_hours)
    rows = con.execute(
        """
        SELECT valid_time, actual, base_lower, base_upper, served_lower, served_upper
        FROM (
            SELECT f.valid_time,
                   pr.power_mw AS actual,
                   coalesce(oi.base_lower, f.p10) AS base_lower,
                   coalesce(oi.base_upper, f.p90) AS base_upper,
                   f.p10 AS served_lower,
                   f.p90 AS served_upper,
                   row_number() OVER (
                       PARTITION BY f.valid_time
                       ORDER BY f.issue_time DESC, f.created_at DESC
                   ) AS rn
            FROM forecasts f
            JOIN production pr
              ON pr.plant_id=f.plant_id AND pr.ts=f.valid_time
            LEFT JOIN online_conformal_issued oi
              ON oi.plant_id=f.plant_id
             AND oi.horizon_hours=f.horizon_hours
             AND oi.valid_time=f.valid_time
            LEFT JOIN online_conformal_feedback fb
              ON fb.plant_id=f.plant_id
             AND fb.horizon_hours=f.horizon_hours
             AND fb.valid_time=f.valid_time
            WHERE f.plant_id=? AND f.horizon_hours=?
              AND f.valid_time < ? AND fb.valid_time IS NULL
              AND f.p10 IS NOT NULL AND f.p90 IS NOT NULL
        )
        WHERE rn=1
        ORDER BY valid_time
        """,
        [plant_id, horizon_hours, issue_time],
    ).fetchall()
    for valid_time, actual, base_lo, base_hi, served_lo, served_hi in rows:
        score, error = state.update(
            actual=float(actual),
            base_lower=float(base_lo),
            base_upper=float(base_hi),
            served_lower=float(served_lo),
            served_upper=float(served_hi),
        )
        con.execute(
            """INSERT INTO online_conformal_feedback
               (plant_id, horizon_hours, valid_time, actual, score, error, state_version)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT (plant_id, horizon_hours, valid_time) DO NOTHING""",
            [
                plant_id,
                horizon_hours,
                valid_time,
                float(actual),
                score,
                error,
                state.feedback_count,
            ],
        )
        if last_valid_time is None or valid_time > last_valid_time:
            last_valid_time = valid_time
    if rows or not state_exists:
        _save_state(
            con,
            plant_id=plant_id,
            horizon_hours=horizon_hours,
            state=state,
            last_feedback_valid_time=last_valid_time,
        )
    return state.adjust(pred), state.feedback_count


def record_issued_band(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    valid_time: dt.datetime,
    issue_time: dt.datetime,
    point_id: int,
    base_lower: float,
    base_upper: float,
    served_lower: float,
    served_upper: float,
    model_id: int | None,
    strategy_trial_id: int | None,
    state_version: int,
) -> None:
    """Persist raw/served interval lineage without changing replayed rows."""
    con.execute(
        """INSERT INTO online_conformal_issued
           (plant_id, horizon_hours, valid_time, issue_time, point_id,
            base_lower, base_upper, served_lower, served_upper, model_id,
            strategy_trial_id, state_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT (plant_id, horizon_hours, valid_time) DO NOTHING""",
        [
            plant_id,
            horizon_hours,
            valid_time,
            issue_time,
            point_id,
            base_lower,
            base_upper,
            served_lower,
            served_upper,
            model_id,
            strategy_trial_id,
            state_version,
        ],
    )
