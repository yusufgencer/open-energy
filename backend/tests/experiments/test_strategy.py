import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from openenergy.calibration.conformal import CQRCalibrator
from openenergy.experiments.strategy import (
    ENSEMBLE_METHODS,
    FittedCandidate,
    QUANTILE_LEVELS,
    QUANTILE_POLICIES,
    StrategyArtifact,
    combine_bands,
    fit_calibrator,
    fit_qra_meta,
    fit_shared_weight,
    fit_vincentization,
    linear_pool_quantiles,
    optimize_weights,
    pinball_loss,
    p50_regime_bin_edges,
    select_perquantile_ensemble,
    select_quantile_policy,
    total_pinball,
    weights_from_cv,
)
from openenergy.models.base import Prediction


def _asymmetric_members(rng, m=200):
    """Two members with asymmetric per-quantile skill.

    Member A (index 0) nails the true p10 but its p90 is biased wide-high;
    member B (index 1) nails p90 but its p10 is biased wide-low. Neither
    single-weight combination can be best at both tails simultaneously.
    """
    mu = rng.normal(0.0, 1.0, m)
    sigma = 1.0
    z = 1.2816  # ~10th/90th normal quantile
    y = mu + rng.normal(0.0, sigma, m)
    tp10, tp50, tp90 = mu - z * sigma, mu, mu + z * sigma
    a = {0.1: tp10, 0.5: tp50, 0.9: tp90 + 1.5}
    b = {0.1: tp10 - 1.5, 0.5: tp50, 0.9: tp90}
    q_mats = {t: np.vstack([a[t], b[t]]) for t in (0.1, 0.5, 0.9)}
    return q_mats, y


def test_strategy_combine_enforces_monotone_bands():
    # Aggregated member bands can cross the ensemble p50; combine must fix it.
    art = StrategyArtifact(candidates=[], ensemble_method="mean")
    p50s = [np.array([5.0, 5.0])]
    p10s = [np.array([6.0, 1.0])]  # first entry crosses above p50
    p90s = [np.array([2.0, 9.0])]  # first entry crosses below p50
    pred = art._combine(p50s, p10s, p90s)
    assert np.array_equal(pred.p50, [5.0, 5.0])  # p50 unchanged
    assert np.all(pred.p10 <= pred.p50)
    assert np.all(pred.p50 <= pred.p90)


def test_strategy_combine_without_bands_returns_p50_only():
    art = StrategyArtifact(candidates=[], ensemble_method="mean")
    pred = art._combine([np.array([1.0, 2.0])], [], [])
    assert pred.p10 is None and pred.p90 is None
    assert np.array_equal(pred.p50, [1.0, 2.0])


def test_combine_bands_legacy_averages_only_banded_members():
    # D1-t3: legacy methods combine bands as the plain mean over the members that
    # actually produced a band; a bandless member (all-NaN row) contributes nothing.
    p10s = np.vstack([np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.full(2, np.nan)])
    p90s = np.vstack([np.array([5.0, 6.0]), np.array([7.0, 8.0]), np.full(2, np.nan)])
    p10, p90 = combine_bands("mean", None, p10s, p90s)
    assert np.allclose(p10, [2.0, 3.0])   # mean of members 0 and 1 only
    assert np.allclose(p90, [6.0, 7.0])


def test_combine_bands_all_members_bandless_returns_none():
    nan = np.full((2, 3), np.nan)
    p10, p90 = combine_bands("mean", None, nan, nan)
    assert p10 is None and p90 is None


def test_combine_bands_perquantile_applies_learned_weights():
    # For vincentization/qra_meta each band is combined with its own learned weight
    # vector when it aligns with the banded members.
    qw = {0.1: np.array([1.0, 0.0]), 0.5: np.array([0.5, 0.5]), 0.9: np.array([0.0, 1.0])}
    p10s = np.vstack([np.array([0.0, 1.0]), np.array([-5.0, -5.0])])
    p90s = np.vstack([np.array([9.0, 9.0]), np.array([2.0, 3.0])])
    p10, p90 = combine_bands("vincentization", qw, p10s, p90s)
    assert np.allclose(p10, [0.0, 1.0])   # member A only
    assert np.allclose(p90, [2.0, 3.0])   # member B only


def test_combine_bands_perquantile_falls_back_to_mean_when_misaligned():
    # A bandless member drops out so the learned weight vector no longer aligns
    # with the banded members -> mirror serving and fall back to the plain mean.
    qw = {0.1: np.array([1.0, 0.0]), 0.5: np.array([0.5, 0.5]), 0.9: np.array([0.0, 1.0])}
    p10s = np.vstack([np.array([0.0, 2.0]), np.full(2, np.nan)])
    p90s = np.vstack([np.array([8.0, 9.0]), np.full(2, np.nan)])
    p10, p90 = combine_bands("qra_meta", qw, p10s, p90s)
    assert np.allclose(p10, [0.0, 2.0])   # single banded member -> its own band
    assert np.allclose(p90, [8.0, 9.0])


def test_weights_from_cv_prefers_lower_error_and_normalizes():
    weights = weights_from_cv([0.1, 0.2, 0.4])
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_optimize_weights_returns_simplex_weights():
    y = np.array([1.0, 2.0, 3.0])
    p50s = np.vstack([
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
    ])
    weights = optimize_weights(p50s, y)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert weights[0] > weights[1]


def test_strategy_artifact_stores_target_policy_and_geometry(tmp_path):
    from openenergy.features.target import PlantGeometry

    geom = PlantGeometry(lat=39.9, lon=32.8, tilt=5.0, azimuth=180.0)
    art = StrategyArtifact(candidates=[], ensemble_method="none",
                           target_policy="kpv", geometry=geom)
    path = tmp_path / "art.pkl"
    art.save(path)
    loaded = StrategyArtifact.load(path)
    assert loaded.target_policy == "kpv"
    assert loaded.geometry == geom


def test_strategy_artifact_defaults_capacity_norm():
    art = StrategyArtifact(candidates=[], ensemble_method="mean")
    assert art.target_policy == "capacity_norm"
    assert art.geometry is None


def test_ensemble_methods_include_perquantile_entries():
    assert "vincentization" in ENSEMBLE_METHODS
    assert "linear_pool" in ENSEMBLE_METHODS
    assert "qra_meta" in ENSEMBLE_METHODS
    # mean must remain listed before the learned methods (champion baseline).
    assert ENSEMBLE_METHODS.index("mean") < ENSEMBLE_METHODS.index("vincentization")
    assert ENSEMBLE_METHODS.index("mean") < ENSEMBLE_METHODS.index("qra_meta")


def test_linear_pool_never_narrower_property():
    """Linear CDF pooling retains between-member spread unlike quantile averaging."""
    rng = np.random.default_rng(20260718)
    for _ in range(200):
        n_members = int(rng.integers(2, 9))
        n_rows = int(rng.integers(1, 25))
        p50s = rng.normal(0.0, 2.0, size=(n_members, n_rows))
        p10s = p50s - rng.lognormal(-0.2, 0.7, size=(n_members, n_rows))
        p90s = p50s + rng.lognormal(-0.2, 0.7, size=(n_members, n_rows))
        weights = np.ones(n_members) / n_members

        pooled = linear_pool_quantiles(p10s, p50s, p90s, weights=weights)
        vincent_width = np.mean((weights @ p90s) - (weights @ p10s))
        linear_width = np.mean(pooled.p90 - pooled.p10)

        assert linear_width + 1e-10 >= vincent_width


def test_linear_pool_weighted_mixture_quantiles_are_not_quantile_averages():
    # 25% point mass at 0 and 75% at 10: the mixture median is 10, whereas
    # averaging member medians would incorrectly return 7.5.
    knots = np.array([[0.0], [10.0]])
    pred = linear_pool_quantiles(
        knots, knots, knots, weights=np.array([0.25, 0.75])
    )
    assert np.allclose(pred.p10, [0.0], atol=1e-12)
    assert np.allclose(pred.p50, [10.0], atol=1e-12)
    assert np.allclose(pred.p90, [10.0], atol=1e-12)


def test_linear_pool_offline_and_artifact_paths_are_identical():
    from openenergy.experiments.runner import _strategy_prediction

    p10s = np.array([[-3.0, -1.0], [1.0, 3.0]])
    p50s = np.array([[-2.0, 0.0], [2.0, 4.0]])
    p90s = np.array([[-1.0, 1.0], [3.0, 5.0]])
    offline, weights, *_ = _strategy_prediction(
        "linear_pool",
        p50s,
        [0.1, 0.1],
        p50s,
        np.array([0.0, 0.0]),
        p10s,
        p90s,
        p10s,
        p90s,
    )
    artifact = StrategyArtifact(
        candidates=[], ensemble_method="linear_pool", weights=weights
    )
    served = artifact._combine(list(p50s), list(p10s), list(p90s))

    np.testing.assert_allclose(served.p10, offline.p10)
    np.testing.assert_allclose(served.p50, offline.p50)
    np.testing.assert_allclose(served.p90, offline.p90)


def test_linear_pool_missing_bands_is_safe_and_bandless():
    p50s = np.array([[1.0, 2.0], [3.0, 4.0]])
    nan_bands = np.full_like(p50s, np.nan)
    pred = linear_pool_quantiles(nan_bands, p50s, nan_bands)
    np.testing.assert_allclose(pred.p50, [2.0, 3.0])
    assert pred.p10 is None and pred.p90 is None


def test_pinball_loss_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_vincentization_recovers_best_member_per_quantile():
    rng = np.random.default_rng(0)
    q_mats, y = _asymmetric_members(rng)
    w = fit_vincentization(q_mats, y, QUANTILE_LEVELS, shrink=0.05)
    # Weights are per-quantile: p10 should lean on member A, p90 on member B.
    assert np.argmax(w[0.1]) == 0
    assert np.argmax(w[0.9]) == 1
    for t in QUANTILE_LEVELS:
        assert np.all(w[t] >= 0.0)
        assert np.isclose(w[t].sum(), 1.0)


def test_perquantile_weights_beat_single_weight_on_pinball():
    rng = np.random.default_rng(1)
    q_mats, y = _asymmetric_members(rng)
    taus = QUANTILE_LEVELS
    pq = fit_vincentization(q_mats, y, taus, shrink=0.05)
    single = fit_shared_weight(q_mats, y, taus)
    assert total_pinball(pq, q_mats, y, taus) < total_pinball(single, q_mats, y, taus)


def test_qra_meta_beats_single_weight_on_pinball():
    rng = np.random.default_rng(3)
    q_mats, y = _asymmetric_members(rng)
    taus = QUANTILE_LEVELS
    pq = fit_qra_meta(q_mats, y, taus, shrink=0.1)
    single = fit_shared_weight(q_mats, y, taus)
    assert total_pinball(pq, q_mats, y, taus) < total_pinball(single, q_mats, y, taus)
    for t in taus:
        assert np.all(pq[t] >= 0.0)


def test_mean_preferred_when_learned_does_not_validate():
    # Regime switch: member 0 is accurate on the fit window and biased on the
    # holdout window; member 1 is the reverse. Weights learned on the fit
    # window overfit and lose to the plain mean out-of-sample -> fall back.
    rng = np.random.default_rng(2)
    m = 200
    mu = rng.normal(0.0, 1.0, m)
    sigma, z = 1.0, 1.2816
    y = mu + rng.normal(0.0, sigma, m)
    tp10, tp50, tp90 = mu - z * sigma, mu, mu + z * sigma
    good0 = np.arange(m) < 120  # member 0 good on first 60%

    def member(bad_mask, shift=2.0):
        b = np.where(bad_mask, shift, 0.0)
        return {0.1: tp10 + b, 0.5: tp50 + b, 0.9: tp90 + b}

    m0 = member(~good0)  # bad on holdout tail
    m1 = member(good0)   # bad on fit head
    q_mats = {t: np.vstack([m0[t], m1[t]]) for t in QUANTILE_LEVELS}

    weights, chose_learned = select_perquantile_ensemble(
        "vincentization", q_mats, y, QUANTILE_LEVELS, holdout_frac=0.4
    )
    assert chose_learned is False
    for t in QUANTILE_LEVELS:
        assert np.allclose(weights[t], [0.5, 0.5], atol=1e-9)


def test_learned_chosen_when_it_validates():
    rng = np.random.default_rng(4)
    q_mats, y = _asymmetric_members(rng, m=300)
    weights, chose_learned = select_perquantile_ensemble(
        "vincentization", q_mats, y, QUANTILE_LEVELS, holdout_frac=0.4
    )
    assert chose_learned is True
    assert np.argmax(weights[0.1]) == 0
    assert np.argmax(weights[0.9]) == 1


def test_vincentization_artifact_serves_perquantile_bands():
    qw = {0.1: np.array([1.0, 0.0]), 0.5: np.array([0.5, 0.5]), 0.9: np.array([0.0, 1.0])}
    art = StrategyArtifact(candidates=[], ensemble_method="vincentization",
                           quantile_weights=qw)
    p50s = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    p10s = [np.array([0.0, 1.0]), np.array([-5.0, -5.0])]
    p90s = [np.array([9.0, 9.0]), np.array([2.0, 3.0])]
    pred = art._combine(p50s, p10s, p90s)
    assert np.allclose(pred.p50, [2.0, 3.0])       # 0.5*A + 0.5*B
    assert np.allclose(pred.p10, [0.0, 1.0])       # member A only
    assert np.allclose(pred.p90, [2.0, 3.0])       # member B only
    assert np.all(pred.p10 <= pred.p50) and np.all(pred.p50 <= pred.p90)


def test_predict_from_candidate_arrays_perquantile_p50():
    qw = {0.5: np.array([0.25, 0.75])}
    art = StrategyArtifact(candidates=[], ensemble_method="qra_meta", quantile_weights=qw)
    p50s = np.vstack([np.array([0.0, 0.0]), np.array([4.0, 8.0])])
    assert np.allclose(art.predict_from_candidate_arrays(p50s), [3.0, 6.0])


def test_perquantile_artifact_roundtrip(tmp_path):
    qw = {0.1: np.array([0.7, 0.3]), 0.5: np.array([0.5, 0.5]), 0.9: np.array([0.2, 0.8])}
    art = StrategyArtifact(candidates=[], ensemble_method="vincentization",
                           quantile_weights=qw)
    path = tmp_path / "art.pkl"
    art.save(path)
    loaded = StrategyArtifact.load(path)
    assert loaded.ensemble_method == "vincentization"
    for t in QUANTILE_LEVELS:
        assert np.allclose(loaded.quantile_weights[t], qw[t])


# --------------------------------------------------------------------------- #
# D2-t2/T-21 — offline quantile_policy calibration axis {native, cqr}
# --------------------------------------------------------------------------- #


def test_quantile_policies_axis_members():
    assert QUANTILE_POLICIES == ["native", "cqr"]


def test_aci_policy_removed():
    rng = np.random.default_rng(99)
    y, _p50, lo, hi = _miscalibrated_bands(rng, n=40)
    with pytest.raises(ValueError, match="quantile_policy"):
        fit_calibrator("aci", y, lo, hi)


def _miscalibrated_bands(rng, n=400, half_width=0.3):
    """Symmetric bands that are far too NARROW: nominal 80% but ~covering little."""
    y = rng.normal(0.0, 1.0, n)
    p50 = np.zeros(n)
    lower = p50 - half_width
    upper = p50 + half_width
    return y, p50, lower, upper


def test_fit_calibrator_native_is_identity():
    rng = np.random.default_rng(0)
    y, _p50, lo, hi = _miscalibrated_bands(rng)
    assert fit_calibrator("native", y, lo, hi) is None


def test_fit_calibrator_cqr_widens_narrow_band():
    rng = np.random.default_rng(1)
    y, _p50, lo, hi = _miscalibrated_bands(rng)
    cal = fit_calibrator("cqr", y, lo, hi, alpha=0.20)
    clo, chi = cal.adjust(lo, hi)
    # narrow band gets widened toward the nominal 80% coverage
    assert np.all(chi - clo > hi - lo)
    from openenergy.evaluation.metrics import coverage
    assert abs(coverage(y, clo, chi) - 0.8) < 0.1


def test_select_quantile_policy_picks_cqr_and_improves_coverage():
    rng = np.random.default_rng(2)
    # Disjoint validation slices from the same under-covered process.
    y_cal, _p50c, lo_c, hi_c = _miscalibrated_bands(rng, n=500)
    y_sel, _p50s, lo_s, hi_s = _miscalibrated_bands(rng, n=500)
    from openenergy.evaluation.metrics import coverage

    policy, cal = select_quantile_policy(
        calibration_lower=lo_c, calibration_upper=hi_c, y_calibration=y_cal,
        selection_lower=lo_s, selection_upper=hi_s, y_selection=y_sel, alpha=0.20,
    )
    assert policy == "cqr"
    assert cal is not None
    clo, chi = cal.adjust(lo_s, hi_s)
    native_cov = coverage(y_sel, lo_s, hi_s)
    cal_cov = coverage(y_sel, clo, chi)
    # calibrated coverage is closer to the nominal 0.8 than native's
    assert abs(cal_cov - 0.8) < abs(native_cov - 0.8)


def test_select_quantile_policy_keeps_native_when_already_calibrated():
    rng = np.random.default_rng(3)
    # bands already ~80% covering: native should be kept (tie/no improvement)
    n = 600
    y_cal = rng.normal(0.0, 1.0, n)
    z = 1.2816
    lo_c, hi_c = -z * np.ones(n), z * np.ones(n)
    y_sel = rng.normal(0.0, 1.0, n)
    lo_s, hi_s = -z * np.ones(n), z * np.ones(n)
    policy, cal = select_quantile_policy(
        calibration_lower=lo_c, calibration_upper=hi_c, y_calibration=y_cal,
        selection_lower=lo_s, selection_upper=hi_s, y_selection=y_sel, alpha=0.20,
    )
    assert policy == "native"
    assert cal is None


def test_select_quantile_policy_native_when_no_bands():
    policy, cal = select_quantile_policy(
        calibration_lower=None, calibration_upper=None, y_calibration=np.array([1.0]),
        selection_lower=None, selection_upper=None, y_selection=np.array([1.0]),
    )
    assert policy == "native" and cal is None


def test_select_quantile_policy_signature_has_no_test_args():
    params = inspect.signature(select_quantile_policy).parameters
    assert "y_test" not in params
    assert "test_lower" not in params
    assert "test_upper" not in params


def test_select_quantile_policy_uses_validation_selection_holdout():
    rng = np.random.default_rng(33)
    y_cal, _p50, lo_cal, hi_cal = _miscalibrated_bands(rng, n=500)
    # The selection truth is concentrated inside the native band. A CQR correction
    # fitted on the earlier narrow-band process is therefore needlessly wide and
    # must lose under the proper Winkler interval score.
    y_sel = np.zeros(200)
    lo_sel = np.full(200, -0.3)
    hi_sel = np.full(200, 0.3)

    policy, cal = select_quantile_policy(
        calibration_lower=lo_cal,
        calibration_upper=hi_cal,
        y_calibration=y_cal,
        selection_lower=lo_sel,
        selection_upper=hi_sel,
        y_selection=y_sel,
    )

    assert policy == "native"
    assert cal is None


def test_calibrated_artifact_reproduces_widths_at_serving():
    """The stored calibrator is applied in _combine so serving reproduces the
    calibrated (widened) bands, and it survives a pickle roundtrip."""
    rng = np.random.default_rng(4)
    y, _p50, lo, hi = _miscalibrated_bands(rng, n=400)
    cal = fit_calibrator("cqr", y, lo, hi, alpha=0.20)

    art = StrategyArtifact(candidates=[], ensemble_method="mean",
                           quantile_policy="cqr", calibrator=cal)
    p50s = [np.array([0.0, 0.0])]
    p10s = [np.array([-0.3, -0.3])]
    p90s = [np.array([0.3, 0.3])]
    pred = art._combine(p50s, p10s, p90s)
    # calibration widened the band symmetrically beyond the raw 0.6 width
    assert np.all(pred.p90 - pred.p10 > 0.6)
    assert np.all(pred.p10 <= pred.p50) and np.all(pred.p50 <= pred.p90)

    import pickle
    reloaded = pickle.loads(pickle.dumps(art))
    pred2 = reloaded._combine(p50s, p10s, p90s)
    assert np.allclose(pred2.p10, pred.p10) and np.allclose(pred2.p90, pred.p90)


def test_binned_calibrator_serves_without_regime_arg(monkeypatch):
    """Serving derives the Mondrian regime from combined p50 and applies per-bin Q."""
    regime = np.r_[np.full(20, 0.25), np.full(20, 0.75)]
    lower, upper = regime - 0.1, regime + 0.1
    y = np.r_[regime[:20] + 0.2, regime[20:] + 1.1]
    cal = CQRCalibrator(alpha=0.20, bin_edges=[0.5]).fit(
        y, lower, upper, regime=regime
    )

    class _Banded:
        def predict(self, X):
            p50 = X[:, 0]
            return Prediction(p50=p50, p10=p50 - 0.1, p90=p50 + 0.1)

    candidate = FittedCandidate(
        config=SimpleNamespace(
            point_policy="single_point", target_policy="capacity_norm"
        ),
        model=_Banded(),
        feature_blocks=[],
        feature_names=[],
        cv_nrmse=0.1,
    )
    monkeypatch.setattr(
        "openenergy.experiments.strategy.forecast_matrix",
        lambda *args, **kwargs: (
            np.array([[0.25], [0.75]]),
            np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        ),
    )
    art = StrategyArtifact(
        candidates=[candidate],
        ensemble_method="none",
        quantile_policy="cqr",
        calibrator=cal,
    )

    pred, _ = art.predict_serving(
        None,
        point_id=1,
        kind="wind",
        issue_time=None,
        horizon_hours=24,
        capacity_mw=1.0,
    )

    widths = pred.p90 - pred.p10
    assert widths[1] > widths[0]
    assert np.allclose(
        widths,
        0.2 + 2.0 * cal.bin_adjustment(np.array([0.25, 0.75])),
    )


def test_p50_regime_edges_fall_back_for_constant_predictions():
    assert p50_regime_bin_edges(np.ones(20)) is None


def test_artifact_defaults_quantile_policy_native():
    art = StrategyArtifact(candidates=[], ensemble_method="mean")
    assert art.quantile_policy == "native"
    assert art.calibrator is None
