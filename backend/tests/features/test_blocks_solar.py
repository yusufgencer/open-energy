import polars as pl
from openenergy.features import blocks
from openenergy.providers.openmeteo import variables as V


def _solar_df():
    return pl.DataFrame({
        "valid_time": [None, None],
        "power_mw": [0.0, 5.0],
        "shortwave_radiation": [0.0, 800.0],
        "terrestrial_solar_radiation": [0.0, 1000.0],
        "global_tilted_irradiance": [0.0, 850.0],
        "cloud_cover": [100.0, 10.0],
        "temperature_2m": [10.0, 30.0],
    })


def test_clear_sky_index_night_zero_and_day_ratio():
    out = blocks.clear_sky_index(_solar_df())
    assert out["clear_sky_index"][0] == 0.0
    assert abs(out["clear_sky_index"][1] - 0.8) < 1e-6


def test_temperature_derating():
    out = blocks.temperature_derating(_solar_df())
    cell = 30.0 + 0.035 * 850.0
    assert abs(out["cell_temp"][1] - cell) < 1e-6
    assert abs(out["pv_derate"][1] - (1 - 0.004 * (cell - 25))) < 1e-6


def test_solar_vars_extended():
    assert "terrestrial_solar_radiation" in V.SOLAR_VARS
    assert "cloud_cover_high" in V.SOLAR_VARS


def test_solar_blocks_noop_when_missing():
    df = pl.DataFrame({"valid_time": [None], "power_mw": [1.0]})
    assert blocks.clear_sky_index(df).columns == df.columns
    assert blocks.gti(df).columns == df.columns


def test_clear_sky_index_zero_when_toa_zero_with_sensor_noise():
    # TOA=0 (nighttime) but GHI has small sensor noise value — must yield 0.0, not ~1.5
    df = pl.DataFrame({
        "shortwave_radiation": [0.1],
        "terrestrial_solar_radiation": [0.0],
    })
    out = blocks.clear_sky_index(df)
    assert out["clear_sky_index"][0] == 0.0
