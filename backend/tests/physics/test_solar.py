import numpy as np
import pandas as pd
import pytest

from openenergy.physics.solar import clear_sky_power, daylight_mask

# Istanbul-ish coordinates; summer solstice full UTC day.
LAT, LON = 41.0, 29.0
TILT, AZIMUTH = 30.0, 180.0
CAPACITY = 10.0


def _summer_day():
    return pd.date_range("2024-06-21 00:00", "2024-06-21 23:00", freq="h", tz="UTC")


def test_returns_series_aligned_to_times():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    assert isinstance(p, pd.Series)
    assert len(p) == len(times)
    assert p.index.equals(times)


def test_night_is_zero():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    # Deep night hours (UTC) at this longitude are fully dark.
    night = p.loc[["2024-06-21 00:00", "2024-06-21 01:00", "2024-06-21 22:00", "2024-06-21 23:00"]]
    assert (night == 0.0).all()


def test_never_exceeds_capacity_and_nonnegative():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    assert (p >= 0.0).all()
    assert (p <= CAPACITY + 1e-9).all()


def test_peaks_near_solar_noon():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    peak_hour = int(p.idxmax().hour)
    # Solar noon at lon 29E is ~10:04 UTC; peak must land midday, not morning/evening.
    assert 9 <= peak_hour <= 12
    assert p.max() > 0.0


def test_daytime_power_is_positive():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    assert p.loc["2024-06-21 10:00"] > 0.5 * CAPACITY


def test_capacity_scales_linearly():
    times = _summer_day()
    p1 = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, 1.0)
    p10 = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, 10.0)
    np.testing.assert_allclose(p10.to_numpy(), 10.0 * p1.to_numpy(), rtol=1e-9, atol=1e-9)


def test_zero_capacity_gives_zero():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, 0.0)
    assert (p == 0.0).all()


def test_accepts_naive_index_as_utc():
    times = _summer_day()
    naive = times.tz_localize(None)
    p_naive = clear_sky_power(naive, LAT, LON, TILT, AZIMUTH, CAPACITY)
    p_aware = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY)
    np.testing.assert_allclose(p_naive.to_numpy(), p_aware.to_numpy(), rtol=1e-9, atol=1e-9)


def test_accepts_sequence_of_timestamps():
    times = _summer_day()
    p = clear_sky_power(list(times), LAT, LON, TILT, AZIMUTH, CAPACITY)
    assert isinstance(p, pd.Series)
    assert len(p) == len(times)


def test_empty_times_returns_empty_series():
    p = clear_sky_power(pd.DatetimeIndex([], tz="UTC"), LAT, LON, TILT, AZIMUTH, CAPACITY)
    assert isinstance(p, pd.Series)
    assert len(p) == 0


def test_winter_peak_lower_than_summer():
    summer = _summer_day()
    winter = pd.date_range("2024-12-21 00:00", "2024-12-21 23:00", freq="h", tz="UTC")
    ps = clear_sky_power(summer, LAT, LON, TILT, AZIMUTH, CAPACITY)
    pw = clear_sky_power(winter, LAT, LON, TILT, AZIMUTH, CAPACITY)
    # Northern hemisphere: winter clear-sky ceiling is below summer.
    assert pw.max() < ps.max()


def test_negative_capacity_raises():
    times = _summer_day()
    with pytest.raises(ValueError):
        clear_sky_power(times, LAT, LON, TILT, AZIMUTH, -1.0)


# --- daylight_mask (C1-t4) ---


def test_daylight_mask_returns_bool_array_aligned():
    times = _summer_day()
    mask = daylight_mask(times, LAT, LON)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape[0] == len(times)


def test_daylight_mask_night_hours_false():
    times = _summer_day()
    mask = pd.Series(daylight_mask(times, LAT, LON), index=times)
    # Deep night hours (UTC) at this longitude are dark → excluded.
    night = mask.loc[["2024-06-21 00:00", "2024-06-21 01:00",
                      "2024-06-21 22:00", "2024-06-21 23:00"]]
    assert not night.any()


def test_daylight_mask_midday_true():
    times = _summer_day()
    mask = pd.Series(daylight_mask(times, LAT, LON), index=times)
    assert bool(mask.loc["2024-06-21 10:00"])


def test_daylight_mask_agrees_with_clear_sky_zero_at_night():
    times = _summer_day()
    p = clear_sky_power(times, LAT, LON, TILT, AZIMUTH, CAPACITY).to_numpy()
    mask = daylight_mask(times, LAT, LON)
    # Every zero-power (fully dark) row is excluded by the mask; the mask is
    # stricter than the horizon so it never keeps a dark row.
    assert not mask[p == 0.0].any()


def test_daylight_mask_stricter_threshold_excludes_more():
    times = _summer_day()
    loose = daylight_mask(times, LAT, LON, max_zenith=89.0)
    strict = daylight_mask(times, LAT, LON, max_zenith=70.0)
    # A stricter (smaller) max zenith keeps fewer rows.
    assert strict.sum() <= loose.sum()
    assert strict.sum() < len(times)


def test_daylight_mask_empty_times():
    mask = daylight_mask(pd.DatetimeIndex([], tz="UTC"), LAT, LON)
    assert isinstance(mask, np.ndarray)
    assert mask.shape[0] == 0
    assert mask.dtype == bool
