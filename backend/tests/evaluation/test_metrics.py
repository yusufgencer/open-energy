import numpy as np
import pytest
from openenergy.evaluation import metrics


def test_nrmse_perfect_is_zero():
    y = np.array([0.1, 0.5, 0.9])
    assert metrics.nrmse(y, y) == 0.0


def test_nrmse_known_value():
    y = np.array([0.0, 0.0])
    p = np.array([0.2, 0.0])
    assert abs(metrics.nrmse(y, p) - (0.2 / np.sqrt(2))) < 1e-12


def test_nmae_and_bias():
    y = np.array([0.0, 1.0])
    p = np.array([0.2, 0.6])
    assert abs(metrics.nmae(y, p) - 0.3) < 1e-12
    assert abs(metrics.bias(y, p) - (-0.1)) < 1e-12


def test_pinball_asymmetric():
    y = np.array([1.0])
    p = np.array([0.0])
    assert abs(metrics.pinball_loss(y, p, 0.9) - 0.9) < 1e-12


def test_interval_coverage_and_width():
    y = np.array([0.2, 0.5, 1.2, 1.8])
    lo = np.array([0.0, 0.4, 1.0, 2.0])
    hi = np.array([0.3, 0.6, 1.4, 3.0])
    assert metrics.coverage(y, lo, hi) == 0.75
    assert abs(metrics.interval_width(lo, hi) - 0.475) < 1e-12


def test_winkler_score_penalises_misses():
    y = np.array([0.5, 2.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    # widths: 1 + 1; second row misses by 1 and pays 2/0.2 = 10
    assert metrics.winkler_score(y, lo, hi, alpha=0.2) == 6.0


def test_clopper_pearson_interval_is_exact_and_contains_observed_rate():
    lower, upper = metrics.clopper_pearson_interval(16, 20, confidence=0.95)
    assert lower == pytest.approx(0.563386, abs=1e-6)
    assert upper == pytest.approx(0.942666, abs=1e-6)
    assert lower <= 0.8 <= upper
    assert metrics.clopper_pearson_interval(0, 20)[0] == 0.0
    assert metrics.clopper_pearson_interval(20, 20)[1] == 1.0


def test_coverage_backtests_report_calibration_and_serial_clustering():
    # Exactly 80% coverage makes Kupiec's POF statistic zero.
    hits = np.array([True] * 8 + [False] * 2)
    y = np.where(hits, 0.0, 2.0)
    lo, hi = np.full(10, -1.0), np.full(10, 1.0)
    pof, pof_p = metrics.kupiec_pof_test(
        y, lo, hi, target_coverage=0.8
    )
    assert pof == pytest.approx(0.0)
    assert pof_p == pytest.approx(1.0)

    # A longer sequence with all misses in one cluster preserves 80% marginal
    # coverage but violates independence.
    clustered_hits = np.array([True] * 40 + [False] * 10)
    clustered_y = np.where(clustered_hits, 0.0, 2.0)
    clustered_lo, clustered_hi = np.full(50, -1.0), np.full(50, 1.0)
    lr_ind, ind_p = metrics.christoffersen_independence_test(
        clustered_y, clustered_lo, clustered_hi
    )
    lr_cc, cc_p = metrics.christoffersen_conditional_coverage_test(
        clustered_y, clustered_lo, clustered_hi, target_coverage=0.8
    )
    assert lr_ind > 0.0 and ind_p < 0.05
    assert lr_cc >= lr_ind and cc_p < 0.05


def test_skill_score():
    assert abs(metrics.skill_score(0.5, 1.0) - 0.5) < 1e-12
    assert metrics.skill_score(0.5, 0.0) == 0.0  # guard


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.nrmse(np.array([1.0]), np.array([1.0, 2.0]))


def test_peak_metrics_over_masked_hours():
    y = np.array([1.0, 1.0, 5.0, 5.0])
    p = np.array([1.0, 1.0, 3.0, 4.0])  # underpredicts the peak hours
    mask = np.array([False, False, True, True])
    assert abs(metrics.peak_bias(y, p, mask) - (-1.5)) < 1e-12  # mean((3-5),(4-5))
    assert abs(metrics.peak_mae(y, p, mask) - 1.5) < 1e-12


def test_peak_metrics_empty_mask_returns_none():
    y = np.array([1.0, 2.0])
    p = np.array([1.0, 2.0])
    mask = np.array([False, False])
    assert metrics.peak_bias(y, p, mask) is None
    assert metrics.peak_mae(y, p, mask) is None


def test_crps_approx_prefers_tighter_correct_quantiles():
    y = np.array([0.5, 0.5])
    p50 = np.array([0.5, 0.5])
    good = metrics.crps_approx(y, p50, p10=np.array([0.4, 0.4]), p90=np.array([0.6, 0.6]))
    bad = metrics.crps_approx(y, p50, p10=np.array([0.0, 0.0]), p90=np.array([1.0, 1.0]))
    assert good < bad


def test_crps_approx_p50_only_is_mae():
    y = np.array([0.0, 1.0])
    p = np.array([0.5, 0.5])
    assert abs(metrics.crps_approx(y, p) - metrics.nmae(y, p)) < 1e-12


def test_crps_piecewise_matches_analytic_gaussian():
    """Three Gaussian quantiles give a close piecewise approximation to exact CRPS."""
    from scipy.stats import norm

    y = np.array([0.0, 0.5, 1.0, 2.0])
    p10 = np.full_like(y, norm.ppf(0.1))
    p50 = np.full_like(y, norm.ppf(0.5))
    p90 = np.full_like(y, norm.ppf(0.9))
    piecewise = metrics.crps_values(y, p50, p10, p90)
    analytic = (
        y * (2.0 * norm.cdf(y) - 1.0)
        + 2.0 * norm.pdf(y)
        - 1.0 / np.sqrt(np.pi)
    )
    np.testing.assert_allclose(piecewise, analytic, atol=0.05, rtol=0.0)


def test_dm_test_zero_differential_not_significant():
    """Identical loss series -> no evidence either way, high p-value."""
    d = np.zeros(50)
    stat, p = metrics.dm_test(d, h=1)
    assert p > 0.05
    assert stat == 0.0


def test_dm_test_noisy_zero_mean_not_significant():
    """Zero-mean noise in loss differential is not a significant improvement."""
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 1.0, 200)
    stat, p = metrics.dm_test(d, h=1)
    assert p > 0.05


def test_dm_test_positive_differential_is_significant():
    """A consistent positive differential (challenger better) is significant."""
    rng = np.random.default_rng(1)
    d = 1.0 + rng.normal(0.0, 0.3, 200)  # mean well above zero
    stat, p = metrics.dm_test(d, h=1)
    assert stat > 0
    assert p < 0.01


def test_dm_test_sign_follows_mean():
    rng = np.random.default_rng(2)
    d = -1.0 + rng.normal(0.0, 0.3, 200)  # challenger worse
    stat, p = metrics.dm_test(d, h=1)
    assert stat < 0
    assert p < 0.01


def test_dm_test_requires_two_points():
    with pytest.raises(ValueError):
        metrics.dm_test(np.array([1.0]), h=1)


def test_dm_test_requires_horizon():
    with pytest.raises(TypeError):
        metrics.dm_test(np.array([0.0, 1.0]))


def test_block_ci_covers_true_metric_on_iid_data():
    """The dependent-data CI stays near nominal on stationary AR(1) errors."""
    covered = 0
    simulations = 50
    phi = 0.45
    innovation_scale = np.sqrt(1.0 - phi**2)
    for simulation_seed in range(simulations):
        rng = np.random.default_rng(simulation_seed)
        errors = np.empty(260)
        errors[0] = rng.normal()
        for t in range(1, errors.size):
            errors[t] = phi * errors[t - 1] + rng.normal(scale=innovation_scale)
        lower, upper = metrics.block_bootstrap_ci(
            errors**2,
            statistic=lambda squared: float(np.sqrt(np.mean(squared))),
            confidence=0.95,
            n_resamples=500,
            block_length=8,
            seed=simulation_seed,
        )
        covered += lower <= 1.0 <= upper
    assert covered / simulations >= 0.90


def test_block_bootstrap_is_deterministic_and_supports_stationary():
    values = np.arange(30, dtype=float)
    first = metrics.block_bootstrap_ci(
        values, n_resamples=100, block_length=4, seed=17, method="stationary"
    )
    second = metrics.block_bootstrap_ci(
        values, n_resamples=100, block_length=4, seed=17, method="stationary"
    )
    assert first == second


def test_model_confidence_set_eliminates_worse_and_keeps_equivalents():
    rng = np.random.default_rng(4)
    common = rng.normal(size=240) ** 2
    losses = np.column_stack([common, common.copy(), common + 0.5])
    members = metrics.model_confidence_set(
        losses, n_resamples=500, block_length=5, seed=9
    )
    assert members == [0, 1]


def test_nine_day_single_origin_is_exploratory():
    assert metrics.evaluation_grade(n_days=9, origins=1, seasons=1) == "exploratory"


def test_evaluation_grade_requires_every_dimension_for_higher_tier():
    assert metrics.evaluation_grade(n_days=30, origins=2, seasons=2) == "limited"
    assert metrics.evaluation_grade(n_days=365, origins=3, seasons=4) == "robust"
    assert metrics.evaluation_grade(n_days=730, origins=1, seasons=4) == "exploratory"
