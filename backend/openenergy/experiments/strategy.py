from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from openenergy.calibration.conformal import (
    CQRCalibrator,
)
from openenergy.evaluation import metrics
from openenergy.experiments.search_space import QUANTILE_POLICIES
from openenergy.features.target import PlantGeometry, kpv_inverse
from openenergy.forecasting.assembly import forecast_matrix, forecast_matrix_multipoint
from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order

# Ordered so `mean` (the champion baseline) is always evaluated before the
# learned per-quantile combiners — on ties the earlier method wins, keeping
# `mean` preferred when learned weights do not genuinely improve (D1-t2).
ENSEMBLE_METHODS = [
    "none",
    "mean",
    "weighted_by_cv",
    "optuna_weights",
    "stacking_meta_ridge",
    "vincentization",
    "linear_pool",
    "qra_meta",
]

# Learned per-quantile combiners that carry a `quantile_weights` dict on the
# artifact (one weight vector per quantile level) instead of a single p50 weight.
PERQUANTILE_METHODS = ("vincentization", "qra_meta")

# The quantile levels combined by the per-quantile machinery. p50 is always
# present; p10/p90 may be missing for candidates without bands (masked, not
# zero-filled) and are combined only over the candidates that produced them.
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
LINEAR_POOL_METHOD = "linear_pool"


@dataclass
class FittedCandidate:
    config: object
    model: ModelWrapper
    feature_blocks: list[str]
    feature_names: list[str]
    cv_nrmse: float
    test_nrmse: float | None = None


@dataclass
class StrategyConfig:
    candidates: list[object]
    ensemble_method: str
    top_k: int


@dataclass
class StrategyArtifact:
    candidates: list[FittedCandidate]
    ensemble_method: str
    weights: np.ndarray | None = None
    meta_model: Ridge | None = None
    meta_scaler: StandardScaler | None = None
    # Per-quantile combination weights (D1-t2), one non-negative vector per
    # quantile level, aligned with ``candidates`` order. Set only for the
    # PERQUANTILE_METHODS ("vincentization" / "qra_meta"); None for every legacy
    # method so their combine path stays byte-identical.
    quantile_weights: dict[float, np.ndarray] | None = None
    # Serving-parity fields (B2-t5): the ordered grid point set and the NWP
    # source policy used to ASSEMBLE the training raw frame. Each candidate's
    # own config carries point_policy/source_policy; these let serving rebuild
    # the exact multipoint/multi-source transform. None → legacy single-point
    # serving (byte-identical to before).
    point_ids: list[int] | None = None
    source_policy: str | None = None
    # kPV target policy (C1-t3). When target_policy == "kpv", candidate models
    # predict the clear-sky ratio ŷ' and serving inverts ŷ' × P_cs; ``geometry``
    # is the reference used to rebuild P_cs at serving time. Defaults preserve the
    # legacy capacity_norm serving (byte-identical).
    target_policy: str = "capacity_norm"
    geometry: "PlantGeometry | None" = None
    # Conformal calibration (D2-t2). ``quantile_policy`` is the searched, remembered
    # offline calibration axis {native, cqr}; ``calibrator`` is the fitted state (a
    # CQRCalibrator, or None for "native") applied to the combined ensemble band in
    # :meth:`_combine` so serving reproduces the calibrated widths seen at eval time.
    # Defaults preserve the legacy native (uncalibrated) path byte-identically.
    quantile_policy: str = "native"
    calibrator: "CQRCalibrator | None" = None

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "StrategyArtifact":
        return pickle.loads(Path(path).read_bytes())

    def predict_from_candidate_arrays(self, p50s: np.ndarray) -> np.ndarray:
        if p50s.ndim != 2:
            raise ValueError("p50s shape must be (n_candidates, n_rows)")
        if self.ensemble_method == "none":
            return p50s[0]
        if self.ensemble_method in {"mean", "weighted_by_cv", "optuna_weights"}:
            weights = self.weights
            if weights is None:
                weights = np.ones(p50s.shape[0], dtype=float) / p50s.shape[0]
            return weights @ p50s
        if self.ensemble_method in PERQUANTILE_METHODS:
            return self._weighted_quantile(0.5, p50s)
        if self.ensemble_method == LINEAR_POOL_METHOD:
            # A linear pool's point forecast is the median of the mixture
            # distribution when bands are available. In p50-only contexts use
            # the artifact weights as a deterministic point fallback.
            rows = np.asarray(p50s, dtype=float)
            weights = self.weights
            if weights is None or np.asarray(weights).shape != (rows.shape[0],):
                weights = np.ones(rows.shape[0], dtype=float) / rows.shape[0]
            return _apply_weights(np.asarray(weights, dtype=float), rows)
        if self.ensemble_method == "stacking_meta_ridge":
            assert self.meta_model is not None and self.meta_scaler is not None
            return self.meta_model.predict(self.meta_scaler.transform(p50s.T))
        raise ValueError(f"bilinmeyen ensemble_method: {self.ensemble_method}")

    def _weighted_quantile(self, tau: float, rows: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Combine one quantile level across candidates with its learned weights.

        Falls back to a plain mean when no per-quantile weight vector exists for
        ``tau`` or when it does not align with the number of member rows (e.g. a
        serving call where some candidate dropped out) — never crossing into a
        different member ordering.
        """
        M = np.vstack(rows) if isinstance(rows, list) else rows
        w = None if self.quantile_weights is None else self.quantile_weights.get(tau)
        if w is None or np.asarray(w).shape[0] != M.shape[0]:
            return np.mean(M, axis=0)
        return _apply_weights(np.asarray(w, dtype=float), M)

    def _combine(
        self,
        p50s: list[np.ndarray],
        p10s: list[np.ndarray],
        p90s: list[np.ndarray],
    ) -> Prediction:
        """Combine candidate predictions into one non-crossing Prediction.

        For legacy methods p50 is combined via the fitted ensemble method and the
        p10/p90 bands are averaged across the candidates that produced them. For
        the per-quantile methods each quantile is combined with its own learned
        weight vector. The result always passes through enforce_quantile_order so
        the aggregation can never emit a band that crosses the ensemble p50.
        """
        if self.ensemble_method == LINEAR_POOL_METHOD:
            # Callers preserve member alignment with NaN rows for bandless
            # candidates. The shared implementation is also used by offline
            # validation/test evaluation.
            lower = _aligned_band_matrix(p10s, len(p50s))
            upper = _aligned_band_matrix(p90s, len(p50s))
            pred = linear_pool_quantiles(
                lower,
                np.vstack(p50s),
                upper,
                weights=self.weights,
            )
            return self._apply_calibration(pred)
        if self.ensemble_method in PERQUANTILE_METHODS and self.quantile_weights is not None:
            p50 = self._weighted_quantile(0.5, p50s)
            p10 = self._weighted_quantile(0.1, p10s) if p10s else None
            p90 = self._weighted_quantile(0.9, p90s) if p90s else None
            return self._apply_calibration(
                enforce_quantile_order(Prediction(p50=p50, p10=p10, p90=p90))
            )
        p50 = self.predict_from_candidate_arrays(np.vstack(p50s))
        p10 = np.mean(np.vstack(p10s), axis=0) if p10s else None
        p90 = np.mean(np.vstack(p90s), axis=0) if p90s else None
        return self._apply_calibration(
            enforce_quantile_order(Prediction(p50=p50, p10=p10, p90=p90))
        )

    def _apply_calibration(self, pred: Prediction) -> Prediction:
        """Apply the stored conformal calibrator to a combined band (D2-t2).

        Widens/tightens the ensemble p10/p90 by the calibrator's additive
        correction, then re-enforces non-crossing ordering. A native (None)
        calibrator or a bandless prediction passes through unchanged, keeping the
        legacy path byte-identical.
        """
        cal = getattr(self, "calibrator", None)
        if cal is None or pred.p10 is None or pred.p90 is None:
            return pred
        # The combined p50 is the canonical Mondrian regime both offline and at
        # serving. The artifact's stored bin_edges therefore suffice; callers do
        # not need to reconstruct or pass a separate weather-regime feature.
        regime = (
            np.asarray(pred.p50, dtype=float)
            if getattr(cal, "bin_edges", None) is not None
            else None
        )
        lo, hi = cal.adjust(
            np.asarray(pred.p10, dtype=float),
            np.asarray(pred.p90, dtype=float),
            regime=regime,
        )
        return enforce_quantile_order(Prediction(p50=pred.p50, p10=lo, p90=hi))

    def _to_norm_space(self, cand, pred: Prediction, valid_times: np.ndarray,
                       capacity_mw: float) -> Prediction:
        """Map one candidate's raw output into the COMMON normalized-power space.

        A kpv candidate predicts the clear-sky ratio ŷ'; invert it through the
        physical envelope P_cs and divide by capacity so it shares a single scale
        with the capacity_norm members before combination (T-01). Combining the
        raw ŷ'/ŷ across spaces over-predicts on ramps where P_cs << capacity.
        This mirrors the offline ``_norm_prediction`` used to rank candidates so
        serving and evaluation agree. capacity_norm passes through unchanged.
        """
        cand_tp = getattr(cand.config, "target_policy", "capacity_norm")
        if cand_tp != "kpv" or self.geometry is None:
            return pred
        from openenergy.physics.solar import clear_sky_power

        g = self.geometry
        p_cs = clear_sky_power(
            valid_times, g.lat, g.lon, g.tilt, g.azimuth, capacity_mw
        ).to_numpy()

        def inv(a):
            return kpv_inverse(a, p_cs, capacity_mw) / capacity_mw

        return Prediction(
            p50=inv(pred.p50),
            p10=inv(pred.p10) if pred.p10 is not None else None,
            p90=inv(pred.p90) if pred.p90 is not None else None,
        )

    def predict_serving(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        point_id: int,
        kind: str,
        issue_time,
        horizon_hours: int,
        capacity_mw: float,
    ) -> tuple[Prediction, np.ndarray]:
        p50s: list[np.ndarray] = []
        p10s: list[np.ndarray] = []
        p90s: list[np.ndarray] = []
        valid_times: np.ndarray | None = None

        for cand in self.candidates:
            cand_point_policy = getattr(cand.config, "point_policy", "single_point")
            if self.point_ids and cand_point_policy != "single_point":
                # Grid champion: rebuild the exact multipoint/multi-source transform.
                X, vt = forecast_matrix_multipoint(
                    con,
                    point_ids=self.point_ids,
                    kind=kind,
                    feature_blocks=cand.feature_blocks,
                    feature_names=cand.feature_names,
                    issue_time=issue_time,
                    horizon_hours=horizon_hours,
                    point_policy=cand_point_policy,
                    source_policy=self.source_policy,
                )
            else:
                X, vt = forecast_matrix(
                    con,
                    point_id=point_id,
                    kind=kind,
                    feature_blocks=cand.feature_blocks,
                    feature_names=cand.feature_names,
                    issue_time=issue_time,
                    horizon_hours=horizon_hours,
                )
            if X.shape[0] == 0:
                continue
            # Normalize each candidate to the common normalized-power space BEFORE
            # combining, so kpv (ŷ') and capacity_norm (ŷ) members never mix (T-01).
            pred = self._to_norm_space(cand, cand.model.predict(X), vt, capacity_mw)
            p50s.append(pred.p50)
            if self.ensemble_method == LINEAR_POOL_METHOD:
                # Linear pooling needs each member's three knots kept aligned.
                # A member without a complete band is masked and excluded from
                # the mixture (rather than silently becoming a point mass).
                p10s.append(
                    pred.p10 if pred.p10 is not None else np.full_like(pred.p50, np.nan)
                )
                p90s.append(
                    pred.p90 if pred.p90 is not None else np.full_like(pred.p50, np.nan)
                )
            else:
                if pred.p10 is not None:
                    p10s.append(pred.p10)
                if pred.p90 is not None:
                    p90s.append(pred.p90)
            valid_times = vt

        if not p50s or valid_times is None:
            return Prediction(p50=np.empty((0,))), np.empty((0,), dtype="datetime64[us]")

        return self._combine(p50s, p10s, p90s), valid_times


def weights_from_cv(cv_scores: list[float]) -> np.ndarray:
    arr = np.asarray(cv_scores, dtype=float)
    inv = 1.0 / np.clip(arr, 1e-9, None)
    return inv / inv.sum()


def optimize_weights(p50s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small deterministic projected random search for non-negative ensemble weights."""
    n = p50s.shape[0]
    if n == 1:
        return np.ones(1)
    rng = np.random.default_rng(42)
    best = np.ones(n) / n
    best_err = float(np.sqrt(np.mean((best @ p50s - y) ** 2)))
    for _ in range(256):
        w = rng.dirichlet(np.ones(n))
        err = float(np.sqrt(np.mean((w @ p50s - y) ** 2)))
        if err < best_err:
            best, best_err = w, err
    return best


# --------------------------------------------------------------------------- #
# Per-quantile ensemble combination (D1-t2)
#
# p10 errors ≠ p90 errors, so a single p50 weight vector is the wrong tool for
# combining the bands. Two learned combiners are added to the pool:
#   * "vincentization" — a convex (simplex) weight vector PER quantile level,
#   * "qra_meta"       — quantile-regression averaging: non-negative coefficients
#                        PER quantile level fit to pinball loss (a pinball-loss
#                        upgrade of the squared-error Ridge stacking meta).
# Both shrink toward uniform, and per the forecast-combination puzzle both must
# beat the plain `mean` on a held-out slice of the validation window before the
# runner will prefer them (see select_perquantile_ensemble).
# --------------------------------------------------------------------------- #


def pinball_loss(y: np.ndarray, pred: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss at level ``tau``."""
    d = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def _apply_weights(w: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Combine member rows ``Q`` (n_members × n_obs) with weights ``w``.

    NaN member rows (candidates without this band) contribute nothing; the
    weight vector is expected to already carry 0 for those members.
    """
    return np.asarray(w, dtype=float) @ np.nan_to_num(Q, nan=0.0)


def _aligned_band_matrix(
    rows: list[np.ndarray] | np.ndarray,
    n_members: int,
) -> np.ndarray:
    """Return a member-aligned band matrix, or an all-NaN mask if ambiguous."""
    if isinstance(rows, np.ndarray):
        matrix = np.asarray(rows, dtype=float)
    elif rows:
        matrix = np.vstack(rows).astype(float, copy=False)
    else:
        return np.empty((n_members, 0), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != n_members:
        # Legacy callers historically dropped bandless members, losing their
        # identity. Do not guess an alignment for a distribution mixture.
        return np.empty((n_members, 0), dtype=float)
    return matrix


def _member_piecewise_cdf(
    x: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> np.ndarray:
    """CDF implied by a constant-tail, piecewise-linear quantile function.

    The representation is identical to ``evaluation.metrics.crps_values``:
    Q(tau) is constant on [0,.1] and [.9,1], and linear between the supplied
    p10/p50/p90 knots. Equal knots are handled as atoms.
    """
    out = np.zeros_like(q10, dtype=float)
    knots = ((q10, q50, 0.1, 0.4), (q50, q90, 0.5, 0.4))
    out += 0.1 * (x >= q10)
    for left, right, _base, mass in knots:
        width = right - left
        continuous = np.divide(
            x - left,
            width,
            out=np.zeros_like(q10, dtype=float),
            where=width > 0.0,
        )
        contribution = mass * np.clip(continuous, 0.0, 1.0)
        contribution = np.where(width > 0.0, contribution, mass * (x >= left))
        out += contribution
    out += 0.1 * (x >= q90)
    return out


def linear_pool_quantiles(
    p10s: np.ndarray,
    p50s: np.ndarray,
    p90s: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> Prediction:
    """Quantiles of a weighted linear mixture of member distributions.

    This combines CDFs and then inverts the mixture; it deliberately does not
    average corresponding quantiles (Vincentization). Members without a
    complete finite p10/p50/p90 triplet are excluded per observation and the
    remaining weights are renormalised. If no complete member exists for an
    observation, the p50 falls back to the mean member point forecast and the
    whole result is returned bandless.
    """
    median = np.asarray(p50s, dtype=float)
    lower = np.asarray(p10s, dtype=float)
    upper = np.asarray(p90s, dtype=float)
    if median.ndim != 2:
        raise ValueError("p50s shape must be (n_members, n_rows)")
    if lower.shape != median.shape or upper.shape != median.shape:
        return Prediction(p50=np.mean(median, axis=0))

    # Canonicalise every member before interpreting it as a quantile function.
    knots = np.sort(np.stack([lower, median, upper], axis=0), axis=0)
    lower, median_ordered, upper = knots
    valid = np.isfinite(lower) & np.isfinite(median_ordered) & np.isfinite(upper)
    if weights is None or np.asarray(weights).shape != (median.shape[0],):
        base_weights = np.ones(median.shape[0], dtype=float)
    else:
        base_weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if not np.isfinite(base_weights).all() or base_weights.sum() <= 0.0:
            base_weights = np.ones(median.shape[0], dtype=float)

    effective = base_weights[:, None] * valid
    sums = effective.sum(axis=0)
    usable = sums > 0.0
    if not usable.all():
        # Prediction cannot express bands only for a subset of rows. Preserve a
        # safe, deterministic point forecast and report no band.
        return Prediction(p50=np.nanmean(np.asarray(p50s, dtype=float), axis=0))
    effective /= sums

    def invert(tau: float) -> np.ndarray:
        lo = np.min(np.where(valid, lower, np.inf), axis=0)
        hi = np.max(np.where(valid, upper, -np.inf), axis=0)
        # Bisection is deterministic and correctly lands on mixture atoms.
        for _ in range(64):
            mid = (lo + hi) / 2.0
            cdf = np.sum(
                effective * _member_piecewise_cdf(mid[None, :], lower, median_ordered, upper),
                axis=0,
            )
            hi = np.where(cdf >= tau, mid, hi)
            lo = np.where(cdf < tau, mid, lo)
        return hi

    return enforce_quantile_order(
        Prediction(p10=invert(0.1), p50=invert(0.5), p90=invert(0.9))
    )


def combine_bands(
    method: str,
    quantile_weights: dict[float, np.ndarray] | None,
    p10s: np.ndarray,
    p90s: np.ndarray,
    p50s: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Combine candidate p10/p90 test bands into one ensemble band per quantile.

    ``p10s``/``p90s`` are ``(n_candidates × n_obs)`` matrices with an all-NaN row
    for every candidate that produced no band (masked, never zero-filled). The
    combination mirrors :meth:`StrategyArtifact._combine` used at serving time so
    ensemble evaluation and serving agree: for the per-quantile methods the
    learned ``quantile_weights`` are applied when they align with the members that
    produced the band, otherwise (and for every legacy method) the band is the
    plain mean over the members that produced it. A quantile with no producing
    member yields ``None`` (no band). Ordering is enforced by the caller.
    """
    if method == LINEAR_POOL_METHOD:
        if p50s is None:
            return None, None
        pred = linear_pool_quantiles(p10s, p50s, p90s, weights=weights)
        return pred.p10, pred.p90

    def _one(tau: float, band: np.ndarray) -> np.ndarray | None:
        if band.size == 0:
            return None
        valid = ~np.isnan(band).any(axis=1)
        if not valid.any():
            return None
        bv = band[valid]
        w = None
        if method in PERQUANTILE_METHODS and quantile_weights is not None:
            w = quantile_weights.get(tau)
        if w is not None and np.asarray(w).shape[0] == bv.shape[0]:
            return _apply_weights(np.asarray(w, dtype=float), bv)
        return np.mean(bv, axis=0)

    return _one(0.1, p10s), _one(0.9, p90s)


def _solve_quantile_lp(blocks: list[tuple[np.ndarray, np.ndarray, float]],
                       n: int, simplex: bool) -> np.ndarray:
    """Minimise summed pinball loss over ``blocks`` with one shared weight vector.

    Each block is ``(Q, y, tau)`` where ``Q`` is (n × m). Returns non-negative
    weights ``b`` (length n); when ``simplex`` they additionally sum to 1. Solved
    as a linear program (residual split into non-negative under/over parts).
    """
    m_sizes = [Q.shape[1] for Q, _, _ in blocks]
    total_m = int(sum(m_sizes))
    n_var = n + 2 * total_m
    c = np.zeros(n_var)
    a_eq = np.zeros((total_m, n_var))
    b_eq = np.zeros(total_m)
    row = 0
    off = n
    for Q, y, tau in blocks:
        m = Q.shape[1]
        Qf = np.nan_to_num(Q, nan=0.0)
        for i in range(m):
            a_eq[row, :n] = Qf[:, i]
            a_eq[row, off + i] = 1.0        # u_i (under)
            a_eq[row, off + m + i] = -1.0   # v_i (over)
            b_eq[row] = y[i]
            c[off + i] = tau
            c[off + m + i] = 1.0 - tau
            row += 1
        off += 2 * m
    if simplex:
        a_eq = np.vstack([a_eq, np.concatenate([np.ones(n), np.zeros(2 * total_m)])])
        b_eq = np.concatenate([b_eq, [1.0]])
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=[(0.0, None)] * n_var, method="highs")
    if not res.success:
        return (np.ones(n) / n) if simplex else np.ones(n) / n
    return np.clip(res.x[:n], 0.0, None)


def _fit_one_quantile(Q: np.ndarray, y: np.ndarray, tau: float, *,
                      simplex: bool, shrink: float) -> np.ndarray:
    """Fit a weight vector for a single quantile, masking bandless members.

    Members whose row is entirely NaN (no band produced) are excluded from the
    fit and receive weight 0 — never zero-filled into the regression. The fitted
    weights are shrunk toward uniform (the forecast-combination regulariser).
    """
    n = Q.shape[0]
    w = np.zeros(n)
    valid = ~np.isnan(Q).any(axis=1)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.ones(n) / n
    if idx.size == 1:
        w[idx[0]] = 1.0
        return w
    b = _solve_quantile_lp([(Q[idx], y, tau)], idx.size, simplex)
    if simplex:
        uni = np.ones(idx.size) / idx.size
        b = (1.0 - shrink) * b + shrink * uni
        s = b.sum()
        b = b / s if s > 0 else uni
    else:
        uni = np.full(idx.size, b.mean() if b.sum() > 0 else 1.0 / idx.size)
        b = (1.0 - shrink) * b + shrink * uni
    w[idx] = b
    return w


def fit_vincentization(q_mats: dict[float, np.ndarray], y: np.ndarray,
                       taus=QUANTILE_LEVELS, shrink: float = 0.1) -> dict[float, np.ndarray]:
    """Per-quantile convex (Vincentization) weights minimising pinball loss."""
    return {t: _fit_one_quantile(q_mats[t], y, t, simplex=True, shrink=shrink) for t in taus}


def fit_qra_meta(q_mats: dict[float, np.ndarray], y: np.ndarray,
                 taus=QUANTILE_LEVELS, shrink: float = 0.1) -> dict[float, np.ndarray]:
    """Quantile-regression-averaging meta: per-quantile non-negative pinball weights."""
    return {t: _fit_one_quantile(q_mats[t], y, t, simplex=False, shrink=shrink) for t in taus}


def fit_shared_weight(q_mats: dict[float, np.ndarray], y: np.ndarray,
                      taus=QUANTILE_LEVELS, simplex: bool = True) -> dict[float, np.ndarray]:
    """Single weight vector shared across every quantile (the single-weight baseline)."""
    n = next(iter(q_mats.values())).shape[0]
    b = _solve_quantile_lp([(q_mats[t], y, t) for t in taus], n, simplex)
    if simplex and b.sum() > 0:
        b = b / b.sum()
    return {t: b for t in taus}


def uniform_weights_by_tau(q_mats: dict[float, np.ndarray],
                           taus=QUANTILE_LEVELS) -> dict[float, np.ndarray]:
    """Uniform weights over the members that produced each band (== plain mean)."""
    out: dict[float, np.ndarray] = {}
    for t in taus:
        Q = q_mats[t]
        n = Q.shape[0]
        valid = ~np.isnan(Q).any(axis=1)
        w = np.zeros(n)
        if valid.any():
            w[valid] = 1.0 / int(valid.sum())
        else:
            w[:] = 1.0 / n
        out[t] = w
    return out


def total_pinball(weights_by_tau: dict[float, np.ndarray], q_mats: dict[float, np.ndarray],
                  y: np.ndarray, taus=QUANTILE_LEVELS) -> float:
    """Summed pinball loss across ``taus`` for a per-quantile weight scheme."""
    return float(sum(
        pinball_loss(y, _apply_weights(weights_by_tau[t], q_mats[t]), t) for t in taus
    ))


def select_perquantile_ensemble(method: str, q_mats: dict[float, np.ndarray], y: np.ndarray,
                                 taus=QUANTILE_LEVELS, holdout_frac: float = 0.4,
                                 shrink: float = 0.1) -> tuple[dict[float, np.ndarray], bool]:
    """Fit per-quantile weights and keep them only if they beat `mean` out-of-sample.

    The validation window is split walk-forward into a fit slice and a trailing
    holdout slice. Weights are learned on the fit slice; if they do not strictly
    beat uniform (the plain `mean`) on the holdout slice, uniform weights are
    returned instead. Returns ``(weights_by_tau, chose_learned)``. When learned
    weights validate, they are refit on the full window before returning.
    """
    fitter = fit_vincentization if method == "vincentization" else fit_qra_meta
    m = len(y)
    n_hold = max(1, int(round(m * holdout_frac)))
    n_fit = m - n_hold
    if n_fit < 2:
        return uniform_weights_by_tau(q_mats, taus), False
    q_fit = {t: q_mats[t][:, :n_fit] for t in taus}
    q_hold = {t: q_mats[t][:, n_fit:] for t in taus}
    y_fit, y_hold = y[:n_fit], y[n_fit:]
    learned = fitter(q_fit, y_fit, taus, shrink)
    uniform = uniform_weights_by_tau(q_fit, taus)
    if total_pinball(learned, q_hold, y_hold, taus) < total_pinball(uniform, q_hold, y_hold, taus):
        return fitter(q_mats, y, taus, shrink), True
    return uniform_weights_by_tau(q_mats, taus), False


# --------------------------------------------------------------------------- #
# Conformal calibration selection (D2-t2)
#
# The offline quantile_policy axis {native, cqr} sits ON TOP of the combined ensemble
# band. A calibrator is FIT on the ensemble's validation-window band and APPLIED to
# the test-window band, so the recorded coverage/width stay honest (the calibrator's
# parameters never see the test truth). The chosen calibrator is stored on the
# StrategyArtifact and re-applied identically at serving via _apply_calibration.
# --------------------------------------------------------------------------- #


def fit_calibrator(
    policy: str,
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
    *,
    regime: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
) -> CQRCalibrator | None:
    """Fit the calibrator for ``policy`` on a validation band, or None for native.

    ``cqr`` fits a split-conformal (CQR) additive width correction. Adaptive
    policies are rejected here: online adaptation is a serving-only terminal
    layer and must never observe validation or outer-test truth.
    """
    if policy == "native":
        return None
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if policy == "cqr":
        return CQRCalibrator(alpha=alpha, bin_edges=bin_edges).fit(
            y, lower, upper, regime=regime if bin_edges is not None else None
        )
    raise ValueError(f"bilinmeyen quantile_policy: {policy}")


def p50_regime_bin_edges(
    p50: np.ndarray, *, n_bins: int = 3
) -> np.ndarray | None:
    """Return stable, data-derived interior edges for p50 Mondrian regimes.

    Quantile edges give each calibration bucket useful support without assuming
    technology-specific absolute power thresholds. Duplicate/non-finite edges
    are removed; a constant p50 naturally falls back to global CQR.
    """
    values = np.asarray(p50, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < n_bins or n_bins < 2:
        return None
    edges = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    edges = edges[(edges > np.min(values)) & (edges < np.max(values))]
    return edges if edges.size else None


def select_quantile_policy(
    *,
    calibration_lower: np.ndarray | None,
    calibration_upper: np.ndarray | None,
    y_calibration: np.ndarray,
    selection_lower: np.ndarray | None,
    selection_upper: np.ndarray | None,
    y_selection: np.ndarray,
    calibration_regime: np.ndarray | None = None,
    selection_regime: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
    alpha: float = 0.20,
    policies: list[str] = QUANTILE_POLICIES,
) -> tuple[str, CQRCalibrator | None]:
    """Choose a quantile policy using validation data only.

    The caller supplies two disjoint, time-ordered validation slices. A candidate
    calibrator is fit on ``calibration_*`` and evaluated on the untouched
    ``selection_*`` slice with the Winkler interval score. The outer test window is
    deliberately absent from this interface and may only be scored after this
    function returns. ``native`` wins ties, so calibration must strictly improve
    the proper validation score.

    Keeping calibration and selection arrays explicit makes the boundary auditable
    now and lets T-18 replace ``calibration_*`` with OOF predictions without
    changing the selection contract.
    """
    if (
        calibration_lower is None
        or calibration_upper is None
        or selection_lower is None
        or selection_upper is None
    ):
        return "native", None

    y_selection = np.asarray(y_selection, dtype=float)
    selection_lower = np.asarray(selection_lower, dtype=float)
    selection_upper = np.asarray(selection_upper, dtype=float)

    best_policy: str = "native"
    best_cal: CQRCalibrator | None = None
    best_score = metrics.winkler_score(
        y_selection, selection_lower, selection_upper, alpha=alpha
    )
    for policy in policies:
        if policy == "native":
            continue
        cal = fit_calibrator(
            policy,
            np.asarray(y_calibration, dtype=float),
            np.asarray(calibration_lower, dtype=float),
            np.asarray(calibration_upper, dtype=float),
            alpha,
            regime=calibration_regime,
            bin_edges=bin_edges if policy == "cqr" else None,
        )
        lo, hi = cal.adjust(
            selection_lower,
            selection_upper,
            regime=(
                selection_regime
                if getattr(cal, "bin_edges", None) is not None
                else None
            ),
        )
        score = metrics.winkler_score(y_selection, lo, hi, alpha=alpha)
        if score < best_score - 1e-12:
            best_policy, best_cal, best_score = policy, cal, score
    return best_policy, best_cal
