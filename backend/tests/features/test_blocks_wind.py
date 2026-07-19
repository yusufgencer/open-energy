import polars as pl
from openenergy.features.blocks import _first_present
from openenergy.features import blocks


def test_first_present_picks_existing():
    df = pl.DataFrame({"wind_speed_100m": [1.0], "valid_time": [None]})
    assert _first_present(df, ["wind_speed_120m", "wind_speed_100m"]) == "wind_speed_100m"
    assert _first_present(df, ["nope"]) is None


def _wind_df():
    return pl.DataFrame({
        "valid_time": [None, None],
        "power_mw": [1.0, 2.0],
        "wind_speed_100m": [3.0, 4.0],
        "wind_speed_10m": [1.5, 2.0],
        "wind_direction_100m": [0.0, 90.0],
        "wind_gusts_10m": [5.0, 6.0],
        "surface_pressure": [1013.0, 1010.0],
        "temperature_2m": [15.0, 20.0],
    })


def test_wind_power_adds_cube():
    out = blocks.wind_power(_wind_df())
    assert out["wind_speed_cube"][1] == 64.0   # 4^3
    assert out["power_mw"][0] == 1.0           # hedef korunur


def test_air_density_and_rho_v3():
    out = blocks.air_density(blocks.wind_power(_wind_df()))
    rho0 = 1013.0 * 100 / (287.05 * (15.0 + 273.15))
    assert abs(out["air_density"][0] - rho0) < 1e-6
    assert abs(out["rho_v3"][0] - rho0 * 27.0) < 1e-6   # 3^3


def test_direction_encoding_unit_circle():
    out = blocks.wind_direction(_wind_df())
    import math
    assert abs(out["wind_dir_cos"][0] - 1.0) < 1e-9     # 0° → cos=1
    assert abs(out["wind_dir_sin"][1] - 1.0) < 1e-9     # 90° → sin=1


def test_blocks_noop_when_columns_missing():
    df = pl.DataFrame({"valid_time": [None], "power_mw": [1.0]})
    assert blocks.wind_power(df).columns == df.columns
    assert blocks.wind_shear(df).columns == df.columns


def test_wind_direction_prefers_120m_over_100m():
    import math
    # 120m column has 0° (cos=1, sin=0); 100m column has 90° (cos=0, sin=1).
    # If 120m is preferred, wind_dir_cos must be ~1.0.
    df = pl.DataFrame({
        "wind_direction_120m": [0.0],
        "wind_direction_100m": [90.0],
    })
    out = blocks.wind_direction(df)
    assert abs(out["wind_dir_cos"][0] - 1.0) < 1e-9
    assert abs(out["wind_dir_sin"][0] - 0.0) < 1e-9
