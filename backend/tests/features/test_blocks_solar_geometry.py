import datetime as dt

import numpy as np
import polars as pl

from openenergy.features import FeatureConfig, build_features
from openenergy.features import blocks
from openenergy.features.registry import available_blocks
from openenergy.physics.solar import solar_geometry_features

# Central Anatolia (Turkey). Solar noon (transit) is ~09:40 UTC here.
_LAT, _LON = 39.0, 35.0

_GEO_COLS = [
    "cos_zenith",
    "solar_elevation",
    "azimuth_sin",
    "azimuth_cos",
    "airmass",
    "extra_radiation",
    "day_length",
    "hours_since_sunrise",
    "hours_to_sunset",
    "is_daylight",
]


def _day_times():
    base = dt.datetime(2025, 6, 21, 0, 0)
    return [base + dt.timedelta(hours=3 * i) for i in range(8)]


def _solar_df():
    return pl.DataFrame({
        "valid_time": _day_times(),
        "power_mw": [0.0] * 8,
    })


def test_solar_geometry_registered_for_solar_only():
    assert "solar_geometry" in available_blocks("solar")
    assert "solar_geometry" not in available_blocks("wind")


def test_features_function_emits_all_columns():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    for col in _GEO_COLS:
        assert col in geo
        assert len(geo[col]) == 8


def test_cos_zenith_higher_near_solar_noon_than_morning():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    cz = np.asarray(geo["cos_zenith"])
    # index 3 -> 09:00 UTC (near solar noon), index 1 -> 03:00 UTC (early morning)
    assert cz[3] > cz[1]
    # solar noon has the maximum cos_zenith over the day
    assert cz[3] == np.max(cz)


def test_night_rows_flagged():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    daylight = np.asarray(geo["is_daylight"])
    elevation = np.asarray(geo["solar_elevation"])
    # 00:00 and 21:00 UTC are night at this location
    assert daylight[0] == 0.0
    assert daylight[-1] == 0.0
    assert elevation[0] <= 0.0
    # midday is daylight
    assert daylight[3] == 1.0
    assert elevation[3] > 0.0


def test_azimuth_unit_circle_and_extraterrestrial_range():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    s = np.asarray(geo["azimuth_sin"])
    c = np.asarray(geo["azimuth_cos"])
    assert np.allclose(s**2 + c**2, 1.0, atol=1e-6)
    extra = np.asarray(geo["extra_radiation"])
    assert np.all(extra > 1300.0) and np.all(extra < 1420.0)


def test_day_length_and_sun_offsets_reasonable():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    day_length = np.asarray(geo["day_length"])
    # June solstice at ~39N: ~14.9h of daylight
    assert np.all(day_length > 8.0) and np.all(day_length < 16.0)
    hss = np.asarray(geo["hours_since_sunrise"])
    hts = np.asarray(geo["hours_to_sunset"])
    # midday: after sunrise and before sunset
    assert hss[3] > 0.0
    assert hts[3] > 0.0


def test_airmass_finite_and_night_filled():
    geo = solar_geometry_features(_day_times(), _LAT, _LON)
    am = np.asarray(geo["airmass"])
    assert np.all(np.isfinite(am))
    # daytime airmass >= 1
    assert am[3] >= 1.0
    # night airmass filled to 0 (no valid geometric path)
    assert am[0] == 0.0


def test_block_adds_columns_when_coords_supplied():
    out = blocks.solar_geometry(_solar_df(), lat=_LAT, lon=_LON)
    for col in _GEO_COLS:
        assert col in out.columns
    assert out["power_mw"].to_list() == [0.0] * 8
    assert out.height == 8


def test_block_noop_without_coords():
    df = _solar_df()
    assert blocks.solar_geometry(df).columns == df.columns
    assert blocks.solar_geometry(df, lat=_LAT).columns == df.columns


def test_block_noop_without_datetime_valid_time():
    df = pl.DataFrame({"valid_time": [None, None], "power_mw": [1.0, 2.0]})
    assert blocks.solar_geometry(df, lat=_LAT, lon=_LON).columns == df.columns


def test_build_features_threads_coords_to_block():
    cfg = FeatureConfig(blocks=["solar_geometry"], lat=_LAT, lon=_LON)
    out = build_features(_solar_df(), "solar", cfg)
    assert "cos_zenith" in out.columns
    # other solar blocks remain unaffected: building without coords no-ops geometry
    cfg2 = FeatureConfig(blocks=["solar_geometry"])
    out2 = build_features(_solar_df(), "solar", cfg2)
    assert "cos_zenith" not in out2.columns
