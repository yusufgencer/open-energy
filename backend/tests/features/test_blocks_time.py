import datetime as dt
import math
import polars as pl
from openenergy.features import blocks


def test_cyclical_time_hour():
    df = pl.DataFrame({"valid_time": [dt.datetime(2024, 1, 1, 6, 0)], "power_mw": [1.0]},
                      schema_overrides={"valid_time": pl.Datetime("us")})
    out = blocks.cyclical_time(df)
    assert abs(out["hour_sin"][0] - math.sin(2 * math.pi * 6 / 24)) < 1e-9
    assert {"hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"} <= set(out.columns)


def test_cyclical_time_noop_without_time():
    df = pl.DataFrame({"power_mw": [1.0]})
    assert blocks.cyclical_time(df).columns == df.columns
