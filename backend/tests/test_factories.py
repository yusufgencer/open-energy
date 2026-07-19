import datetime as dt

import pytest


def test_weather_row_factory_derives_valid_time(weather_row_factory):
    issue_time = dt.datetime(2026, 1, 1, 6)

    row = weather_row_factory(issue_time=issue_time, horizon_hours=24)

    assert row["valid_time"] == issue_time + dt.timedelta(hours=24)
    assert row["lead_hours"] == 24


def test_weather_row_factory_rejects_inconsistent_valid_time(weather_row_factory):
    issue_time = dt.datetime(2026, 1, 1, 6)

    with pytest.raises(ValueError, match="valid_time"):
        weather_row_factory(
            issue_time=issue_time,
            horizon_hours=24,
            valid_time=issue_time + dt.timedelta(hours=23),
        )


def test_forecast_row_factory_derives_valid_time(forecast_row_factory):
    issue_time = dt.datetime(2026, 1, 1, 6)

    row = forecast_row_factory(issue_time=issue_time, horizon_hours=48)

    assert row["valid_time"] == issue_time + dt.timedelta(hours=48)
    assert row["horizon_hours"] == 48
