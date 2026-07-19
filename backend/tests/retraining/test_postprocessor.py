import datetime as dt

import numpy as np
import pytest

from openenergy.assets.repository import Plant, create_plant
from openenergy.models.base import Prediction
from openenergy.retraining.postprocessor import (
    apply_postprocessor,
    fit_postprocessor,
)
from openenergy.storage import connect, init_schema


def _plant(db, capacity=10.0):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=capacity))


def _write_pairs(db, *, base_hour, n, bias, rng, plant_id="wf1", horizon=24, point_id=7):
    """Insert n (forecast, actual) pairs where the stored forecast p50 is the true
    production shifted by a constant `bias` MW. Production is the ground truth."""
    base = dt.datetime(2024, 1, 1) + dt.timedelta(hours=base_hour)
    for i in range(n):
        vt = base + dt.timedelta(hours=i)
        actual = 3.0 + 2.0 * np.sin(i / 6.0) + rng.normal(0, 0.15)
        actual = float(max(actual, 0.0))
        fc = actual + bias  # systematically biased forecast
        issue = vt - dt.timedelta(hours=horizon)
        db.execute(
            "INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
            [plant_id, vt, actual],
        )
        db.execute(
            "INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, "
            "valid_time, p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)",
            [plant_id, point_id, horizon, issue, vt, fc - 1.0, fc, fc + 1.0, None],
        )


def test_fit_absorbs_constant_bias(db):
    """A constant +2 MW bias in stored forecasts is learned and removed on apply."""
    _plant(db)
    rng = np.random.default_rng(0)
    bias = 2.0
    _write_pairs(db, base_hour=0, n=120, bias=bias, rng=rng)

    result = fit_postprocessor(db, plant_id="wf1", horizon_hours=24)
    assert result["fitted"] is True
    assert result["n_samples"] >= 100

    # Apply to a fresh batch of biased forecasts; correction should recover truth.
    truth = 3.0 + 2.0 * np.sin(np.arange(50) / 5.0)
    truth = np.clip(truth, 0.0, None)
    biased = Prediction(p50=truth + bias, p10=truth + bias - 1.0, p90=truth + bias + 1.0)
    corrected = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=biased)

    raw_bias = float(np.mean(biased.p50 - truth))
    new_bias = float(np.mean(corrected.p50 - truth))
    assert abs(raw_bias - bias) < 1e-9
    assert abs(new_bias) < 0.5, f"residual bias {new_bias} not absorbed (raw {raw_bias})"


def test_a_few_daily_fits_absorb_bias(db):
    """Accumulating daily (forecast, actual) pairs over a few fits drives the
    residual bias steadily toward zero."""
    _plant(db)
    rng = np.random.default_rng(1)
    bias = 3.0
    residuals = []
    for day in range(4):
        _write_pairs(db, base_hour=day * 24, n=24, bias=bias, rng=rng)
        fit_postprocessor(db, plant_id="wf1", horizon_hours=24)
        truth = np.clip(4.0 + 2.0 * np.cos(np.arange(24) / 4.0), 0.0, None)
        biased = Prediction(p50=truth + bias)
        corrected = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=biased)
        residuals.append(abs(float(np.mean(corrected.p50 - truth))))
    # After a few daily fits the residual bias is small (well under the injected 3 MW).
    assert residuals[-1] < 0.6, f"residual bias trajectory {residuals}"


def test_identity_without_recent_data(db):
    """No (forecast, actual) pairs → fit is a no-op and apply is the identity."""
    _plant(db)
    result = fit_postprocessor(db, plant_id="wf1", horizon_hours=24)
    assert result["fitted"] is False
    assert result["n_samples"] == 0

    pred = Prediction(p50=np.array([1.0, 2.0, 3.0]),
                      p10=np.array([0.5, 1.5, 2.5]),
                      p90=np.array([1.5, 2.5, 3.5]))
    out = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=pred)
    np.testing.assert_array_equal(out.p50, pred.p50)
    np.testing.assert_array_equal(out.p10, pred.p10)
    np.testing.assert_array_equal(out.p90, pred.p90)


def test_apply_identity_when_never_fitted(db):
    """apply with no stored coefficients returns the input unchanged (identity)."""
    _plant(db)
    pred = Prediction(p50=np.array([5.0, 6.0]))
    out = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=pred)
    np.testing.assert_array_equal(out.p50, pred.p50)
    assert out.p10 is None and out.p90 is None


def test_fit_clears_to_identity_when_data_disappears(db):
    """After a real fit, refitting with no recent data clears the coefficients so
    serving degrades back to identity rather than applying stale corrections."""
    _plant(db)
    rng = np.random.default_rng(2)
    _write_pairs(db, base_hour=0, n=100, bias=2.0, rng=rng)
    assert fit_postprocessor(db, plant_id="wf1", horizon_hours=24)["fitted"] is True

    # Restrict lookback so nothing recent qualifies → identity.
    result = fit_postprocessor(db, plant_id="wf1", horizon_hours=24, lookback_days=0)
    assert result["fitted"] is False
    pred = Prediction(p50=np.array([9.0, 8.0]))
    out = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=pred)
    np.testing.assert_array_equal(out.p50, pred.p50)


def test_apply_preserves_monotonicity_and_nonneg(db):
    """Corrected quantiles are non-crossing and never negative."""
    _plant(db)
    rng = np.random.default_rng(3)
    _write_pairs(db, base_hour=0, n=120, bias=5.0, rng=rng)
    fit_postprocessor(db, plant_id="wf1", horizon_hours=24)
    # Heavily biased near-zero forecast: naive correction could go negative/cross.
    pred = Prediction(p50=np.array([5.0, 5.2, 5.1]),
                      p10=np.array([4.0, 4.1, 4.0]),
                      p90=np.array([6.0, 6.3, 6.2]))
    out = apply_postprocessor(db, plant_id="wf1", horizon_hours=24, pred=pred)
    assert np.all(out.p10 <= out.p50) and np.all(out.p50 <= out.p90)
    assert np.all(out.p10 >= 0.0)


def test_scoped_per_plant_and_horizon(db):
    """Coefficients for one plant/horizon do not leak into another."""
    _plant(db)
    create_plant(db, Plant(plant_id="wf2", name="WF2", kind="wind", capacity_mw=10.0))
    rng = np.random.default_rng(4)
    _write_pairs(db, base_hour=0, n=100, bias=2.0, rng=rng, plant_id="wf1", horizon=24)
    fit_postprocessor(db, plant_id="wf1", horizon_hours=24)
    # wf2 (and horizon 48) never fitted → identity.
    pred = Prediction(p50=np.array([7.0, 7.0]))
    out2 = apply_postprocessor(db, plant_id="wf2", horizon_hours=24, pred=pred)
    np.testing.assert_array_equal(out2.p50, pred.p50)
    out48 = apply_postprocessor(db, plant_id="wf1", horizon_hours=48, pred=pred)
    np.testing.assert_array_equal(out48.p50, pred.p50)


def _calibrated_band(n, rng):
    """A genuinely per-point-calibrated ~80% central band around a heteroskedastic
    Gaussian truth. Returns (p10, p50, p90, actual) arrays."""
    z = 1.2815515594  # 80% central interval half-width in sigmas
    t = np.linspace(0.0, 4.0 * np.pi, n)
    mu = 5.0 + 3.0 * np.sin(t)
    sigma = 0.4 + 0.15 * mu  # heteroskedastic: spread grows with level
    actual = np.clip(mu + rng.normal(0.0, 1.0, n) * sigma, 0.0, None)
    return mu - z * sigma, mu, mu + z * sigma, actual


def test_apply_preserves_band_coverage(db):
    """T-02: post-processing a genuinely calibrated ~80% band must PRESERVE its
    coverage. A squared-error (Lasso) fit regresses every quantile onto the
    conditional mean, collapsing the band toward p50 → coverage far below 80%.
    A proper pinball (quantile-regression) fit keeps each quantile distinct."""
    _plant(db)
    p10, p50, p90, actual = _calibrated_band(400, np.random.default_rng(7))
    base = dt.datetime(2024, 1, 1)
    for i in range(len(p50)):
        vt = base + dt.timedelta(hours=i)
        db.execute("INSERT INTO production (plant_id, ts, power_mw) VALUES (?,?,?)",
                   ["wf1", vt, float(actual[i])])
        db.execute(
            "INSERT INTO forecasts (plant_id, point_id, horizon_hours, issue_time, "
            "valid_time, p10, p50, p90, model_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ["wf1", 7, 24, vt - dt.timedelta(hours=24), vt,
             float(p10[i]), float(p50[i]), float(p90[i]), None],
        )

    assert fit_postprocessor(db, plant_id="wf1", horizon_hours=24)["fitted"] is True

    # Held-out window from the same DGP (arrays only) → apply → measure coverage.
    hp10, hp50, hp90, hactual = _calibrated_band(600, np.random.default_rng(99))
    corrected = apply_postprocessor(
        db, plant_id="wf1", horizon_hours=24,
        pred=Prediction(p50=hp50, p10=hp10, p90=hp90),
    )
    inside = (hactual >= corrected.p10) & (hactual <= corrected.p90)
    coverage = float(np.mean(inside))
    assert 0.75 <= coverage <= 0.85, f"post-processed coverage {coverage:.3f} left [0.75, 0.85]"


def test_old_schema_migrates_postprocessor_table(tmp_path):
    """A persistent DB created before the coefficients table existed gains it on
    the next init_schema, and remains usable."""
    db_file = tmp_path / "old.duckdb"
    con = connect(db_file)
    init_schema(con)
    con.execute("DROP TABLE postprocessor_coefficients")  # simulate an old DB
    con.close()

    con2 = connect(db_file)
    init_schema(con2)  # must recreate the table idempotently
    # Table exists and is writable.
    con2.execute(
        "INSERT INTO postprocessor_coefficients "
        "(plant_id, horizon_hours, quantile, degree, coefficients, n_samples) "
        "VALUES ('p', 24, 'p50', 2, '{}', 10)"
    )
    n = con2.execute("SELECT count(*) FROM postprocessor_coefficients").fetchone()[0]
    assert n == 1
    con2.close()
