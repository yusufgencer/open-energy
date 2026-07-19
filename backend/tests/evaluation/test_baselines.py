import numpy as np
import pytest
from openenergy.evaluation import baselines, metrics


def test_persistence_repeats_last():
    out = baselines.persistence_forecast(np.array([0.1, 0.2, 0.7]), 2)
    assert list(out) == [0.7, 0.7]


def test_climatology_is_mean():
    out = baselines.climatology_forecast(np.array([0.0, 1.0]), 3)
    assert list(out) == [0.5, 0.5, 0.5]


def test_baseline_errors_dict():
    errs = baselines.baseline_errors(np.array([0.2, 0.4]), np.array([0.5, 0.5]), metrics.nrmse)
    assert set(errs) == {"persistence", "climatology"}


def test_empty_train_raises():
    with pytest.raises(ValueError):
        baselines.persistence_forecast(np.array([]), 1)


def _diurnal_series(hours=72):
    base = np.datetime64("2026-01-01T00:00:00")
    vt = np.array([base + np.timedelta64(h, "h") for h in range(hours)])
    y = np.array([float(h % 24) for h in range(hours)])  # depends only on hour-of-day
    return vt, y


def test_diurnal_persistence_matches_same_hour_previous_day():
    vt, y = _diurnal_series(72)
    vt_test = vt[48:]
    out = baselines.diurnal_persistence_forecast(y, vt, vt_test)
    assert np.allclose(out, y[48:])  # perfectly diurnal → same-hour-yesterday is exact


def test_diurnal_persistence_falls_back_when_no_previous_day():
    vt, y = _diurnal_series(6)  # only 6 hours: no t-24h exists
    out = baselines.diurnal_persistence_forecast(y, vt, vt)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))  # graceful fallback, no NaN/crash


def test_baseline_errors_includes_diurnal_when_times_given():
    vt, y = _diurnal_series(72)
    errs = baselines.baseline_errors(
        y[:48], y[48:], metrics.nrmse, vt_train=vt[:48], vt_test=vt[48:]
    )
    assert "diurnal_persistence" in errs
    assert errs["diurnal_persistence"] < errs["persistence"]  # diurnal ~0, flat large
