"""Distribution-free conformal calibration of quantile bands (D2-t1).

Two calibrators sit on top of any model that already emits a lower/upper
quantile band (e.g. post-A1 ordered p10/p90):

* :class:`CQRCalibrator` — Conformalized Quantile Regression (Romano et al.,
  2019). Split-conformal, finite-sample marginal-coverage guarantee. Optionally
  *binned by regime* (solar: clear-sky-index bins; wind: wind-speed bins) so
  each regime gets its own additive width correction and calibrates
  independently.

* :class:`AdaptiveConformalCalibrator` — Adaptive Conformal Inference / ACI
  (Gibbs & Candès, 2021). Runs online over a stream, tracking an effective
  miscoverage level ``alpha_t`` on a rolling calibration window so the interval
  width adapts to distribution drift.

Neither calibrator re-implements quantile-ordering; callers that want ordered
p10/p50/p90 use :meth:`CQRCalibrator.calibrate_prediction`, which delegates the
final clamp to :func:`openenergy.models.base.enforce_quantile_order`.

The sign convention (shared by both) is the CQR conformity score::

    s_i = max(lower_i - y_i, y_i - upper_i)

which is negative when ``y_i`` sits comfortably inside the band, zero on an
edge, and equals the signed distance to the nearest edge when ``y_i`` is
outside. The (1-alpha) empirical quantile ``Q`` of these scores is added
symmetrically to the band: ``[lower - Q, upper + Q]``. ``Q`` may be negative,
which *tightens* an over-wide band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openenergy.models.base import Prediction, enforce_quantile_order

__all__ = [
    "conformity_scores",
    "empirical_coverage",
    "conformal_quantile",
    "CQRCalibrator",
    "AdaptiveConformalCalibrator",
    "CalibratedStream",
]


def conformity_scores(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """CQR conformity score ``max(lower - y, y - upper)`` (element-wise)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.maximum(lower - y_true, y_true - upper)


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of ``y_true`` inside ``[lower, upper]`` (order-insensitive)."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected ``(1-alpha)`` empirical quantile of ``scores``.

    Uses the standard split-conformal level ``ceil((n+1)(1-alpha)) / n`` and the
    ``"higher"`` interpolation so the marginal-coverage guarantee holds. If the
    corrected level exceeds 1 (tiny ``n``) the additive correction is ``+inf``
    (band becomes all-covering), matching the conservative conformal fallback.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 0.0
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        return float("inf")
    if level <= 0.0:
        return float(np.min(scores))
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class CQRCalibrator:
    """Split-conformal (CQR) width correction, optionally binned by regime.

    Parameters
    ----------
    alpha:
        Target miscoverage; nominal coverage is ``1 - alpha`` (0.2 -> 80%).
    bin_edges:
        Interior regime boundaries (ascending). ``None`` (default) fits one
        global correction. With ``k`` interior edges there are ``k+1`` bins,
        assigned by :func:`numpy.digitize`. Each bin gets its own ``Q``; a bin
        that had no calibration samples falls back to the global ``Q``.
    """

    alpha: float = 0.20
    bin_edges: np.ndarray | None = None
    _q_global: float = field(default=0.0, init=False, repr=False)
    _q_by_bin: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_global_q(cls, q: float, alpha: float = 0.20) -> "CQRCalibrator":
        """Build a fitted, unbinned calibrator with a fixed additive correction ``q``.

        Used to freeze an adaptive (ACI) calibrator's resolved width correction into
        a serialisable split-conformal state that can be applied identically at
        serving time.
        """
        cal = cls(alpha=alpha)
        cal._q_global = float(q)
        cal._fitted = True
        return cal

    def __post_init__(self) -> None:
        if self.bin_edges is not None:
            edges = np.asarray(self.bin_edges, dtype=float)
            if edges.ndim != 1 or edges.size == 0:
                raise ValueError("bin_edges must be a non-empty 1-D sequence")
            if np.any(np.diff(edges) <= 0):
                raise ValueError("bin_edges must be strictly ascending")
            self.bin_edges = edges

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        regime: np.ndarray | None = None,
    ) -> "CQRCalibrator":
        scores = conformity_scores(y_true, lower, upper)
        self._q_global = conformal_quantile(scores, self.alpha)
        self._q_by_bin = {}
        if self.bin_edges is not None:
            if regime is None:
                raise ValueError("regime is required when bin_edges is set")
            bins = self._assign_bins(regime)
            for b in range(len(self.bin_edges) + 1):
                sel = bins == b
                if np.any(sel):
                    self._q_by_bin[b] = conformal_quantile(scores[sel], self.alpha)
        self._fitted = True
        return self

    # -- application -------------------------------------------------------
    def adjust(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        regime: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated ``(lower, upper)`` bands (order preserved)."""
        self._require_fitted()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        q = self._row_adjustment(lower.shape, regime)
        lo = lower - q
        hi = upper + q
        # additive symmetric correction preserves ordering, but a negative Q on
        # a degenerate (lower > upper) input could invert it; clamp defensively.
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def bin_adjustment(self, regime: np.ndarray) -> np.ndarray:
        """Per-row additive correction ``Q`` for the given regime values."""
        self._require_fitted()
        regime = np.asarray(regime, dtype=float)
        return self._row_adjustment(regime.shape, regime)

    def calibrate_prediction(
        self, pred: Prediction, regime: np.ndarray | None = None
    ) -> Prediction:
        """Calibrate a :class:`Prediction`'s p10/p90 band; p50 is untouched.

        Ordering is finalised via the canonical
        :func:`enforce_quantile_order` (clamp-to-p50).
        """
        self._require_fitted()
        if pred.p10 is None or pred.p90 is None:
            return pred
        lo, hi = self.adjust(pred.p10, pred.p90, regime=regime)
        return enforce_quantile_order(Prediction(p50=pred.p50, p10=lo, p90=hi))

    # -- internals ---------------------------------------------------------
    def _assign_bins(self, regime: np.ndarray) -> np.ndarray:
        regime = np.asarray(regime, dtype=float)
        return np.digitize(regime, getattr(self, "bin_edges", None))

    def _row_adjustment(
        self, shape: tuple[int, ...], regime: np.ndarray | None
    ) -> np.ndarray:
        # ``getattr`` keeps artifacts pickled before regime-binned calibration
        # readable: their instances do not carry the newer dataclass attributes.
        bin_edges = getattr(self, "bin_edges", None)
        if bin_edges is None:
            return np.full(shape, self._q_global, dtype=float)
        if regime is None:
            raise ValueError("regime is required for a binned calibrator")
        bins = self._assign_bins(regime)
        q = np.full(bins.shape, self._q_global, dtype=float)
        for b, val in getattr(self, "_q_by_bin", {}).items():
            q[bins == b] = val
        return q

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CQRCalibrator must be fit() before use")


@dataclass
class CalibratedStream:
    """Result of :meth:`AdaptiveConformalCalibrator.calibrate_stream`."""

    lower: np.ndarray
    upper: np.ndarray
    alpha_history: np.ndarray


@dataclass
class AdaptiveConformalCalibrator:
    """Adaptive Conformal Inference (ACI) over a stream.

    At each step ``t`` the effective miscoverage ``alpha_t`` picks the conformal
    quantile of scores seen *before* ``t`` (a rolling window of size ``window``,
    or the full history when ``window is None``). After observing whether the
    truth fell in the interval, ``alpha_t`` is nudged::

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where ``err_t = 1`` on a miss. Repeated misses (a drift that narrowed
    coverage) drive ``alpha_t`` down, widening subsequent intervals until
    coverage recovers to the ``1 - alpha`` target.

    ``alpha_t`` is clipped to ``[0, 1]`` for quantile lookup: ``alpha_t <= 0``
    yields an all-covering (+inf width) interval, the intended conformal
    fallback under sustained under-coverage.
    """

    alpha: float = 0.20
    gamma: float = 0.05
    window: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.window is not None and self.window < 1:
            raise ValueError("window must be >= 1")

    def calibrate_stream(
        self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> CalibratedStream:
        y_true = np.asarray(y_true, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        n = y_true.size
        lo_out = np.empty(n, dtype=float)
        hi_out = np.empty(n, dtype=float)
        alpha_hist = np.empty(n, dtype=float)

        history: list[float] = []
        alpha_t = float(self.alpha)
        for t in range(n):
            alpha_hist[t] = alpha_t
            q = self._current_adjustment(history, alpha_t)
            lo_out[t] = lower[t] - q
            hi_out[t] = upper[t] + q

            covered = lo_out[t] <= y_true[t] <= hi_out[t]
            err = 0.0 if covered else 1.0
            alpha_t = alpha_t + self.gamma * (self.alpha - err)
            # keep the tracked level in a sane range so it can recover either way
            alpha_t = float(min(max(alpha_t, -1.0), 2.0))

            history.append(float(conformity_scores(
                np.array([y_true[t]]), np.array([lower[t]]), np.array([upper[t]])
            )[0]))
            if self.window is not None and len(history) > self.window:
                history.pop(0)

        return CalibratedStream(lower=lo_out, upper=hi_out, alpha_history=alpha_hist)

    def _current_adjustment(self, history: list[float], alpha_t: float) -> float:
        if not history:
            return 0.0
        if alpha_t <= 0.0:
            return float("inf")
        if alpha_t >= 1.0:
            # demand ~0% coverage: use the smallest score (tightest correction)
            return float(np.min(history))
        return conformal_quantile(np.asarray(history, dtype=float), alpha_t)
