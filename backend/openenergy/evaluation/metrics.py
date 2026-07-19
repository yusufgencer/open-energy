from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlog1py, xlogy
from typing import Callable


def evaluation_grade(*, n_days: int, origins: int, seasons: int) -> str:
    """Label how much evidence supports an out-of-sample evaluation.

    The grade is deliberately descriptive: callers must not use it as a model
    promotion gate.  All three audit dimensions must meet a tier, so a long
    history evaluated at one origin (or in one season) remains exploratory.
    """
    if n_days < 0 or origins < 0 or seasons < 0:
        raise ValueError("evaluation evidence counts cannot be negative")
    if seasons > 4:
        raise ValueError("seasons cannot exceed four")
    if n_days >= 365 and origins >= 3 and seasons == 4:
        return "robust"
    if n_days >= 30 and origins >= 2 and seasons >= 2:
        return "limited"
    return "exploratory"


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true ve y_pred aynı uzunlukta olmalı")
    if len(y_true) == 0:
        raise ValueError("boş girdi")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    _check(y_true, y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    _check(y_true, lower)
    _check(y_true, upper)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    _check(lower, upper)
    return float(np.mean(np.maximum(lower, upper) - np.minimum(lower, upper)))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = 0.20,
) -> float:
    """Mean Winkler interval score for a central ``1-alpha`` interval.

    Lower is better.  Misses are penalised by their distance from the interval,
    scaled by ``2 / alpha``; crossed bounds are canonicalised row-wise.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 0 ile 1 arasında olmalı")
    _check(y_true, lower)
    _check(y_true, upper)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    score = hi - lo
    score += np.where(y_true < lo, (2.0 / alpha) * (lo - y_true), 0.0)
    score += np.where(y_true > hi, (2.0 / alpha) * (y_true - hi), 0.0)
    return float(np.mean(score))


def clopper_pearson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact finite-sample binomial confidence interval.

    ``successes`` is the number of covered observations.  The boundary cases
    are returned exactly as 0 and 1 instead of relying on beta quantiles with a
    zero shape parameter.
    """
    if n <= 0:
        raise ValueError("n pozitif olmalı")
    if successes < 0 or successes > n:
        raise ValueError("successes 0 ile n arasında olmalı")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 0 ile 1 arasında olmalı")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(stats.beta.ppf(tail, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(
        stats.beta.ppf(1.0 - tail, successes + 1, n - successes)
    )
    return lower, upper


def _interval_hits(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    _check(y_true, lower)
    _check(y_true, upper)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return np.asarray((y_true >= lo) & (y_true <= hi), dtype=bool)


def _binomial_log_likelihood(successes: int, n: int, probability: float) -> float:
    return float(
        xlogy(successes, probability)
        + xlog1py(n - successes, -probability)
    )


def kupiec_pof_test(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_coverage: float = 0.80,
) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood-ratio test.

    Returns ``(LR_pof, p_value)`` against the requested unconditional coverage.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage 0 ile 1 arasında olmalı")
    hits = _interval_hits(y_true, lower, upper)
    n = int(hits.size)
    covered = int(np.sum(hits))
    observed = covered / n
    ll_null = _binomial_log_likelihood(covered, n, target_coverage)
    ll_alt = _binomial_log_likelihood(covered, n, observed)
    statistic = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def christoffersen_independence_test(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    """Christoffersen LR test for serial independence of interval violations."""
    hits = _interval_hits(y_true, lower, upper)
    if hits.size < 2:
        raise ValueError("Christoffersen testi için en az 2 gözlem gerekir")
    violations = ~hits
    previous = violations[:-1]
    current = violations[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    row0 = n00 + n01
    row1 = n10 + n11
    pi01 = n01 / row0 if row0 else 0.0
    pi11 = n11 / row1 if row1 else 0.0
    ll_iid = _binomial_log_likelihood(n01 + n11, total, pi)
    ll_markov = (
        _binomial_log_likelihood(n01, row0, pi01)
        + _binomial_log_likelihood(n11, row1, pi11)
    )
    statistic = max(0.0, -2.0 * (ll_iid - ll_markov))
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def christoffersen_conditional_coverage_test(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_coverage: float = 0.80,
) -> tuple[float, float]:
    """Christoffersen conditional-coverage test (Kupiec POF + independence)."""
    pof, _ = kupiec_pof_test(
        y_true, lower, upper, target_coverage=target_coverage
    )
    independence, _ = christoffersen_independence_test(y_true, lower, upper)
    statistic = pof + independence
    return float(statistic), float(stats.chi2.sf(statistic, df=2))


def peak_bias(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean signed error (pred - true) restricted to peak-hour rows.

    Returns None when the mask selects no rows (e.g. a wind plant, or a solar
    plant with no high-irradiance hours in the window). Diagnoses the systematic
    midday under-prediction that Phase C's kPV transform targets.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(np.mean(y_pred[mask] - y_true[mask]))


def peak_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean absolute error restricted to peak-hour rows; None if mask is empty."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def _integrated_pinball_segment(
    y: float, q_left: float, q_right: float, tau_left: float, tau_right: float
) -> float:
    """Exact integral of pinball loss for a linear quantile segment."""
    width = tau_right - tau_left
    slope = (q_right - q_left) / width
    intercept = q_left - slope * tau_left
    # The pinball expression changes branch where y == Q(tau). Split there so
    # each sub-integral is an ordinary quadratic polynomial.
    cuts = [tau_left, tau_right]
    if slope != 0.0:
        root = (y - intercept) / slope
        if tau_left < root < tau_right:
            cuts.insert(1, root)

    total = 0.0
    c = y - intercept
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = (left + right) / 2.0
        if y >= slope * midpoint + intercept:
            # rho_tau(y-Q) = tau * (c - slope*tau)
            total += (
                0.5 * c * (right ** 2 - left ** 2)
                - (slope / 3.0) * (right ** 3 - left ** 3)
            )
        else:
            # rho_tau(y-Q) = (1-tau) * (slope*tau - c)
            total += (
                -(slope / 3.0) * (right ** 3 - left ** 3)
                + 0.5 * (slope + c) * (right ** 2 - left ** 2)
                - c * (right - left)
            )
    return total


def crps_values(
    y_true: np.ndarray,
    p50: np.ndarray,
    p10: np.ndarray | None = None,
    p90: np.ndarray | None = None,
) -> np.ndarray:
    """Per-observation ``2∫ pinball(tau) d tau`` from p10/p50/p90.

    The quantile function is linear between the supplied levels and constant in
    the two tails. A forecast without a complete p10/p90 band has the explicit,
    method-neutral fallback of a point mass at p50, whose CRPS is absolute error.
    """
    y = np.asarray(y_true, dtype=float)
    median = np.asarray(p50, dtype=float)
    _check(y, median)
    if p10 is None or p90 is None:
        return np.abs(y - median)

    lower = np.asarray(p10, dtype=float)
    upper = np.asarray(p90, dtype=float)
    _check(y, lower)
    _check(y, upper)
    # Invalid crossing forecasts do not define a quantile function. Canonicalise
    # them row-wise, consistently with the prediction pipeline's order guard.
    quantiles = np.sort(np.vstack([lower, median, upper]), axis=0)
    out = np.empty(y.shape[0], dtype=float)
    tau = (0.0, 0.1, 0.5, 0.9, 1.0)
    for i, truth in enumerate(y):
        q10, q50, q90 = quantiles[:, i]
        q = (q10, q10, q50, q90, q90)
        integral = sum(
            _integrated_pinball_segment(
                float(truth), float(q[j]), float(q[j + 1]), tau[j], tau[j + 1]
            )
            for j in range(4)
        )
        out[i] = 2.0 * integral
    return out


def crps_approx(y_true: np.ndarray, p50: np.ndarray,
                p10: np.ndarray | None = None, p90: np.ndarray | None = None) -> float:
    """Mean piecewise-linear quantile CRPS from p10/p50/p90."""
    return float(np.mean(crps_values(y_true, p50, p10, p90)))


def skill_score(model_err: float, ref_err: float) -> float:
    if ref_err <= 1e-12:
        return 0.0
    return 1.0 - model_err / ref_err


def _bootstrap_indices(
    n: int,
    *,
    n_resamples: int,
    block_length: int,
    seed: int,
    method: str,
) -> np.ndarray:
    """Return deterministic, circular block-bootstrap row indices."""
    if n < 2:
        raise ValueError("bootstrap için en az 2 gözlem gerekir")
    if n_resamples < 1:
        raise ValueError("n_resamples pozitif olmalı")
    if block_length < 1 or block_length > n:
        raise ValueError("block_length 1 ile gözlem sayısı arasında olmalı")
    if method not in {"moving", "stationary"}:
        raise ValueError("method 'moving' veya 'stationary' olmalı")

    rng = np.random.default_rng(seed)
    if method == "moving":
        n_blocks = int(np.ceil(n / block_length))
        starts = rng.integers(0, n, size=(n_resamples, n_blocks))
        offsets = np.arange(block_length)
        return ((starts[..., None] + offsets) % n).reshape(n_resamples, -1)[:, :n]

    # Politis-Romano stationary bootstrap. A fresh uniformly selected block
    # starts with probability 1 / block_length; otherwise the circular series
    # continues from the preceding observation.
    out = np.empty((n_resamples, n), dtype=int)
    out[:, 0] = rng.integers(0, n, size=n_resamples)
    restart_probability = 1.0 / block_length
    for t in range(1, n):
        restart = rng.random(n_resamples) < restart_probability
        out[:, t] = np.where(
            restart, rng.integers(0, n, size=n_resamples), (out[:, t - 1] + 1) % n
        )
    return out


def block_bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    block_length: int | None = None,
    seed: int = 0,
    method: str = "moving",
) -> tuple[float, float]:
    """Percentile CI using a deterministic moving/stationary block bootstrap.

    ``values`` are ordered per-timestamp contributions to a metric.  For nRMSE,
    for example, pass squared errors and ``statistic=lambda x:
    sqrt(mean(x))``. Circular blocks keep every replicate the same length and
    avoid privileging observations near either edge.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("values tek boyutlu olmalı")
    if data.size < 2:
        raise ValueError("bootstrap için en az 2 gözlem gerekir")
    if not np.all(np.isfinite(data)):
        raise ValueError("values sonlu olmalı")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 0 ile 1 arasında olmalı")
    length = block_length or max(1, int(round(data.size ** (1.0 / 3.0))))
    indices = _bootstrap_indices(
        data.size,
        n_resamples=n_resamples,
        block_length=length,
        seed=seed,
        method=method,
    )
    estimates = np.fromiter(
        (float(statistic(data[row])) for row in indices),
        dtype=float,
        count=n_resamples,
    )
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def model_confidence_set(
    loss_matrix: np.ndarray,
    *,
    alpha: float = 0.05,
    n_resamples: int = 2_000,
    block_length: int | None = None,
    seed: int = 0,
    bootstrap: str = "moving",
) -> list[int]:
    """Return non-rejected model columns using sequential MCS elimination.

    Rows must be aligned forecast origins and columns competing methods. At each
    step the equal-predictive-ability null is tested with the bootstrapped range
    of centered mean losses; the method with the largest observed mean loss is
    removed only when that null is rejected. The returned column indices retain
    their input order, making the caller's parsimony order a deterministic
    tie-break among statistically indistinguishable methods.
    """
    losses = np.asarray(loss_matrix, dtype=float)
    if losses.ndim != 2:
        raise ValueError("loss_matrix shape (n_rows, n_models) olmalı")
    n, n_models = losses.shape
    if n < 2 or n_models < 1:
        raise ValueError("MCS için en az 2 satır ve 1 model gerekir")
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_matrix sonlu olmalı")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 0 ile 1 arasında olmalı")
    if n_models == 1:
        return [0]

    length = block_length or max(1, int(round(n ** (1.0 / 3.0))))
    indices = _bootstrap_indices(
        n,
        n_resamples=n_resamples,
        block_length=length,
        seed=seed,
        method=bootstrap,
    )
    active = list(range(n_models))
    while len(active) > 1:
        current = losses[:, active]
        observed_means = np.mean(current, axis=0)
        observed_range = float(np.max(observed_means) - np.min(observed_means))
        centered = current - observed_means
        boot_means = np.mean(centered[indices], axis=1)
        boot_ranges = np.max(boot_means, axis=1) - np.min(boot_means, axis=1)
        # Include the observed sample in the Monte-Carlo p-value correction.
        p_value = (1.0 + float(np.sum(boot_ranges >= observed_range))) / (
            n_resamples + 1.0
        )
        if p_value > alpha:
            break
        worst_local = int(np.argmax(observed_means))
        del active[worst_local]
    return active


def dm_test(loss_diff: np.ndarray, *, h: int) -> tuple[float, float]:
    """Diebold-Mariano test on per-timestamp loss differentials.

    ``loss_diff[t] = loss_incumbent[t] - loss_challenger[t]``; a positive mean
    means the challenger has lower loss (is better). Returns ``(dm_stat, p_value)``
    with the Harvey-Leybourne-Newbold small-sample correction and a two-sided
    p-value from Student's t with ``n-1`` degrees of freedom.

    ``h`` is the forecast horizon step for the Newey-West long-run variance
    (autocovariances up to lag ``h-1``); ``h=1`` collapses to the plain variance.
    A degenerate (zero-variance) differential yields ``(0.0, 1.0)`` — no reliable
    test, so the caller must not treat it as significant.
    """
    from scipy import stats

    d = np.asarray(loss_diff, dtype=float)
    n = d.shape[0]
    if d.ndim != 1:
        raise ValueError("loss_diff tek boyutlu olmalı")
    if n < 2:
        raise ValueError("DM testi için en az 2 gözlem gerekir")
    if h < 1:
        raise ValueError("h >= 1 olmalı")
    # The HLN correction needs more observations than the forecast horizon.
    # With n <= h there is no defensible significance test; fail closed so a
    # challenger cannot benefit from an undersized comparison window.
    if h >= n:
        return 0.0, 1.0

    mean_d = float(np.mean(d))
    dev = d - mean_d
    # Newey-West long-run variance estimate with Bartlett weights up to lag h-1.
    var = float(np.mean(dev ** 2))
    for lag in range(1, h):
        # HAC autocovariances use the same ``n`` denominator at every lag.
        cov = float(np.dot(dev[lag:], dev[:-lag]) / n)
        var += 2.0 * (1.0 - lag / h) * cov
    if var <= 0.0:
        return 0.0, 1.0

    dm = mean_d / np.sqrt(var / n)
    # HLN small-sample correction factor.
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm * corr)
    p_value = float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1))
    return dm_hln, p_value
