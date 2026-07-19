"""E1-t3 — Daily light LASSO polynomial quantile post-processor.

HEFTCom pattern: a tiny per-plant+horizon LASSO polynomial quantile regression
fitted daily on recent (forecast, actual) pairs, correcting cheap systematic
drift / capacity bias between the (monthly) base retrains. Stored coefficients
are applied on top of champion forecasts at serving time.

Strictly additive and safe: it is the *identity* whenever there are no recent
pairs to learn from (no coefficients stored, or a refit on empty data clears
them), so a plant that has never been post-processed serves exactly its raw
champion forecast — byte-identical to the pre-E1-t3 path.

Design (kept deliberately small):
  * For each quantile column q in {p10, p50, p90} independently, fit a LASSO on a
    standardised polynomial expansion of the raw forecast value
    ``[f, f^2, ..., f^degree]`` predicting the realised production. Standardising
    each polynomial column (mean/scale, stored) keeps the ill-conditioned f vs
    f^2 scales well-behaved so a light L1 penalty recovers an essentially affine
    de-bias without exploding.
  * Coefficients (degree, per-column mean/scale, intercept, coef) are persisted
    as JSON in ``postprocessor_coefficients`` keyed by (plant_id, horizon, q).
  * ``apply_postprocessor`` maps each quantile through its stored polynomial,
    clips to >= 0, then re-imposes non-crossing order via the canonical helper.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
from sklearn.linear_model import QuantileRegressor

from openenergy.models.base import Prediction, enforce_quantile_order

_QUANTILES = ("p10", "p50", "p90")
# Nominal level each stored column post-processes toward. The fit MUST minimise
# pinball loss at this level (quantile regression), not squared error — a
# squared-error fit collapses every column onto the conditional mean, destroying
# the band's coverage (T-02).
_Q_LEVEL = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
DEFAULT_DEGREE = 2
DEFAULT_ALPHA = 0.001
DEFAULT_MIN_SAMPLES = 12
DEFAULT_LOOKBACK_DAYS = 45

_TABLE = "postprocessor_coefficients"


def _poly(f: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial design matrix ``[f, f^2, ..., f^degree]`` (no intercept column;
    the LASSO fits its own intercept)."""
    f = np.asarray(f, dtype=float)
    return np.column_stack([f ** d for d in range(1, degree + 1)])


def _fit_one(f: np.ndarray, y: np.ndarray, *, degree: int, alpha: float, quantile: float) -> dict:
    """Fit a standardised polynomial quantile regression ``y ~ poly(f)`` at level
    ``quantile`` (pinball loss + L1) and return a serialisable coefficient bundle.

    Pinball loss is essential: a squared-error fit would map every quantile column
    onto the conditional mean, collapsing the band toward p50 (T-02)."""
    X = _poly(f, degree)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)  # guard constant / degenerate columns
    Xs = (X - mean) / scale
    model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=True, solver="highs")
    model.fit(Xs, np.asarray(y, dtype=float))
    return {
        "degree": degree,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(model.intercept_),
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
    }


def _predict_one(f: np.ndarray, bundle: dict) -> np.ndarray:
    degree = int(bundle["degree"])
    mean = np.asarray(bundle["mean"], dtype=float)
    scale = np.asarray(bundle["scale"], dtype=float)
    intercept = float(bundle["intercept"])
    coef = np.asarray(bundle["coef"], dtype=float)
    Xs = (_poly(f, degree) - mean) / scale
    return intercept + Xs @ coef


def _recent_pairs(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    lookback_days: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-quantile ``{q: (forecast, actual)}`` arrays from recent
    (forecast, actual) pairs.

    For each realised valid_time we keep the *latest-issued* forecast at this
    horizon (a valid_time may have been forecast from several issue_times), then
    join against realised production. Only rows whose valid_time is within
    ``lookback_days`` of the most recent realised pair are used, so stale drift
    corrections expire. Quantile columns that are NULL are simply excluded from
    that quantile's fit.
    """
    rows = con.execute(
        f"""
        SELECT valid_time, p10, p50, p90, power_mw FROM (
            SELECT f.valid_time AS valid_time, f.p10 AS p10, f.p50 AS p50,
                   f.p90 AS p90, pr.power_mw AS power_mw,
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
        return {}

    latest = max(r[0] for r in rows)
    cutoff = latest - dt.timedelta(days=lookback_days)
    rows = [r for r in rows if r[0] >= cutoff]
    if not rows:
        return {}

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    col_idx = {"p10": 1, "p50": 2, "p90": 3}
    for q in _QUANTILES:
        idx = col_idx[q]
        fs, ys = [], []
        for r in rows:
            fq = r[idx]
            actual = r[4]
            if fq is None or actual is None:
                continue
            fs.append(float(fq))
            ys.append(float(actual))
        if fs:
            out[q] = (np.asarray(fs, dtype=float), np.asarray(ys, dtype=float))
    return out


def _clear(con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int) -> None:
    con.execute(
        f"DELETE FROM {_TABLE} WHERE plant_id = ? AND horizon_hours = ?",
        [plant_id, horizon_hours],
    )


def fit_postprocessor(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    degree: int = DEFAULT_DEGREE,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict:
    """Fit and persist the post-processor for a plant+horizon from recent pairs.

    Returns ``{"fitted": bool, "n_samples": int, "quantiles": [...]}``. When there
    are fewer than ``min_samples`` recent pairs (or none at all) any existing
    coefficients are cleared and the result is ``fitted=False`` — serving then
    degrades to the identity.
    """
    pairs = _recent_pairs(
        con, plant_id=plant_id, horizon_hours=horizon_hours, lookback_days=lookback_days
    )
    n_samples = max((len(v[0]) for v in pairs.values()), default=0)

    if not pairs or n_samples < min_samples:
        _clear(con, plant_id, horizon_hours)
        return {"fitted": False, "n_samples": 0, "quantiles": []}

    _clear(con, plant_id, horizon_hours)
    fitted_quantiles: list[str] = []
    for q in _QUANTILES:
        if q not in pairs:
            continue
        f, y = pairs[q]
        if len(f) < min_samples:
            continue
        bundle = _fit_one(f, y, degree=degree, alpha=alpha, quantile=_Q_LEVEL[q])
        con.execute(
            f"INSERT INTO {_TABLE} (plant_id, horizon_hours, quantile, degree, "
            f"coefficients, n_samples) VALUES (?,?,?,?,?,?)",
            [plant_id, horizon_hours, q, degree, json.dumps(bundle), len(f)],
        )
        fitted_quantiles.append(q)

    if not fitted_quantiles:
        return {"fitted": False, "n_samples": 0, "quantiles": []}
    return {"fitted": True, "n_samples": n_samples, "quantiles": fitted_quantiles}


def _load(con: duckdb.DuckDBPyConnection, plant_id: str, horizon_hours: int) -> dict[str, dict]:
    rows = con.execute(
        f"SELECT quantile, coefficients FROM {_TABLE} "
        f"WHERE plant_id = ? AND horizon_hours = ?",
        [plant_id, horizon_hours],
    ).fetchall()
    return {q: json.loads(coeffs) for q, coeffs in rows}


def apply_postprocessor(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    horizon_hours: int,
    pred: Prediction,
) -> Prediction:
    """Apply the fitted post-processor to a MW-scale prediction.

    Identity when no coefficients are stored (never-fitted, or cleared). Corrected
    quantiles are clipped to >= 0 and re-ordered non-crossing via the canonical
    ``enforce_quantile_order`` helper — the single source of truth for monotonicity.
    """
    bundles = _load(con, plant_id, horizon_hours)
    if not bundles:
        return pred

    def _corr(arr: np.ndarray | None, q: str) -> np.ndarray | None:
        if arr is None or q not in bundles:
            return arr
        return np.clip(_predict_one(np.asarray(arr, dtype=float), bundles[q]), 0.0, None)

    return enforce_quantile_order(
        Prediction(
            p50=_corr(pred.p50, "p50"),
            p10=_corr(pred.p10, "p10"),
            p90=_corr(pred.p90, "p90"),
        )
    )
