"""Tests for the standalone conformal calibration module (D2-t1).

Coverage claims are checked on synthetic *miscalibrated* bands:
 - global CQR pulls nominal-80% coverage into [0.78, 0.82];
 - regime bins calibrate independently (a global adjustment cannot fix both);
 - ACI adapts its interval width to an injected distribution shift.
"""
from __future__ import annotations

import numpy as np

from openenergy.calibration.conformal import (
    AdaptiveConformalCalibrator,
    CQRCalibrator,
    conformity_scores,
    empirical_coverage,
)
from openenergy.models.base import Prediction


NOMINAL = 0.80
ALPHA = 0.20


def test_conformity_scores_sign_convention() -> None:
    # inside the band -> non-positive score; outside -> positive distance to the
    # nearest edge.
    y = np.array([0.0, 2.0, -2.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    s = conformity_scores(y, lo, hi)
    assert s[0] <= 0.0  # well inside
    assert np.isclose(s[1], 1.0)  # 2.0 is 1.0 above upper edge
    assert np.isclose(s[2], 1.0)  # -2.0 is 1.0 below lower edge


def test_cqr_global_coverage_within_band() -> None:
    """Too-narrow synthetic bands: post-CQR empirical coverage in [0.78, 0.82]."""
    rng = np.random.default_rng(7)
    n_cal, n_test = 3000, 6000
    # Truth ~ N(0, 1); the model emits a deliberately narrow +/-0.3 band
    # (nominal coverage ~24%, far from 80%).
    y_cal = rng.standard_normal(n_cal)
    y_test = rng.standard_normal(n_test)
    lo_cal, hi_cal = np.full(n_cal, -0.3), np.full(n_cal, 0.3)
    lo_test, hi_test = np.full(n_test, -0.3), np.full(n_test, 0.3)

    # sanity: the raw band is badly miscalibrated
    assert empirical_coverage(y_test, lo_test, hi_test) < 0.4

    cal = CQRCalibrator(alpha=ALPHA).fit(y_cal, lo_cal, hi_cal)
    lo_c, hi_c = cal.adjust(lo_test, hi_test)
    cov = empirical_coverage(y_test, lo_c, hi_c)
    assert 0.78 <= cov <= 0.82, cov


def test_cqr_calibrates_from_too_wide_band() -> None:
    """CQR also *tightens* an over-wide band toward nominal (score goes negative)."""
    rng = np.random.default_rng(11)
    n_cal, n_test = 3000, 6000
    y_cal = rng.standard_normal(n_cal)
    y_test = rng.standard_normal(n_test)
    lo_cal, hi_cal = np.full(n_cal, -5.0), np.full(n_cal, 5.0)  # ~100% coverage
    lo_test, hi_test = np.full(n_test, -5.0), np.full(n_test, 5.0)

    cal = CQRCalibrator(alpha=ALPHA).fit(y_cal, lo_cal, hi_cal)
    lo_c, hi_c = cal.adjust(lo_test, hi_test)
    cov = empirical_coverage(y_test, lo_c, hi_c)
    assert 0.78 <= cov <= 0.82, cov
    # tightened: calibrated width strictly smaller than the raw 10.0
    assert (hi_c - lo_c).mean() < 9.0


def test_bins_calibrate_independently() -> None:
    """Two regimes with different noise scale: binned CQR calibrates each bin;
    a single global adjustment cannot (it over-covers one and under-covers the other)."""
    rng = np.random.default_rng(23)

    def make(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # regime encoded by `kc`: low bin sigma=0.5, high bin sigma=3.0
        kc = rng.uniform(0.0, 1.0, n)
        low = kc < 0.5
        sigma = np.where(low, 0.5, 3.0)
        y = rng.standard_normal(n) * sigma
        lo = np.full(n, -0.3)
        hi = np.full(n, 0.3)
        return kc, y, lo, hi

    kc_cal, y_cal, lo_cal, hi_cal = make(6000)
    kc_te, y_te, lo_te, hi_te = make(12000)

    # interior edge at kc = 0.5 -> two bins
    binned = CQRCalibrator(alpha=ALPHA, bin_edges=[0.5]).fit(
        y_cal, lo_cal, hi_cal, regime=kc_cal
    )
    lo_c, hi_c = binned.adjust(lo_te, hi_te, regime=kc_te)

    low_mask = kc_te < 0.5
    high_mask = ~low_mask
    cov_low = empirical_coverage(y_te[low_mask], lo_c[low_mask], hi_c[low_mask])
    cov_high = empirical_coverage(y_te[high_mask], lo_c[high_mask], hi_c[high_mask])
    assert 0.78 <= cov_low <= 0.82, cov_low
    assert 0.78 <= cov_high <= 0.82, cov_high

    # bins really learned different adjustments
    q_low = binned.bin_adjustment(np.array([0.25]))[0]
    q_high = binned.bin_adjustment(np.array([0.75]))[0]
    assert q_high > q_low * 2.0

    # a single global calibrator cannot satisfy both bins simultaneously
    glob = CQRCalibrator(alpha=ALPHA).fit(y_cal, lo_cal, hi_cal)
    lo_g, hi_g = glob.adjust(lo_te, hi_te)
    cov_low_g = empirical_coverage(y_te[low_mask], lo_g[low_mask], hi_g[low_mask])
    cov_high_g = empirical_coverage(y_te[high_mask], lo_g[high_mask], hi_g[high_mask])
    # at least one bin is badly off under the global fit
    assert not (0.78 <= cov_low_g <= 0.82 and 0.78 <= cov_high_g <= 0.82)


def test_cqr_calibrate_prediction_orders_quantiles() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_normal(2000)
    lo, hi = np.full(2000, -0.3), np.full(2000, 0.3)
    cal = CQRCalibrator(alpha=ALPHA).fit(y, lo, hi)
    pred = Prediction(
        p50=np.zeros(5),
        p10=np.full(5, -0.3),
        p90=np.full(5, 0.3),
    )
    out = cal.calibrate_prediction(pred)
    assert np.all(out.p10 <= out.p50)
    assert np.all(out.p50 <= out.p90)
    # widened relative to input
    assert np.all(out.p90 - out.p10 > 0.6)


def test_aci_adapts_to_injected_shift() -> None:
    """A mid-stream shift makes bands too low; ACI must widen/track to recover
    coverage, beating a frozen (static) conformal adjustment."""
    rng = np.random.default_rng(42)
    n = 4000
    shift_at = n // 2
    # stationary phase: y ~ N(0,1); shift phase: y ~ N(+4, 1) (bands too low)
    mean = np.where(np.arange(n) < shift_at, 0.0, 4.0)
    y = rng.standard_normal(n) + mean
    lo = np.full(n, -0.3)
    hi = np.full(n, 0.3)

    aci = AdaptiveConformalCalibrator(alpha=ALPHA, gamma=0.05, window=500)
    res = aci.calibrate_stream(y, lo, hi)

    # look at the tail (post-shift, after adaptation has had time to react)
    tail = slice(shift_at + 500, n)
    cov_tail = empirical_coverage(y[tail], res.lower[tail], res.upper[tail])

    # a static conformal fit frozen on the pre-shift window collapses after shift
    static = CQRCalibrator(alpha=ALPHA).fit(y[:shift_at], lo[:shift_at], hi[:shift_at])
    lo_s, hi_s = static.adjust(lo, hi)
    cov_tail_static = empirical_coverage(y[tail], lo_s[tail], hi_s[tail])

    assert cov_tail_static < 0.5  # frozen fit fails to cover after the shift
    assert cov_tail > 0.65  # adaptive recovers most of nominal coverage
    assert cov_tail > cov_tail_static + 0.2

    # alpha_t should have dropped after the shift (=> wider intervals demanded)
    assert res.alpha_history[shift_at + 300] < ALPHA
    # returned intervals are ordered
    assert np.all(res.lower <= res.upper)


def test_aci_stable_without_shift() -> None:
    """With no shift ACI stays near nominal and doesn't blow up."""
    rng = np.random.default_rng(5)
    n = 3000
    y = rng.standard_normal(n)
    lo, hi = np.full(n, -0.3), np.full(n, 0.3)
    aci = AdaptiveConformalCalibrator(alpha=ALPHA, gamma=0.02, window=400)
    res = aci.calibrate_stream(y, lo, hi)
    cov = empirical_coverage(y[500:], res.lower[500:], res.upper[500:])
    assert 0.72 <= cov <= 0.88, cov
