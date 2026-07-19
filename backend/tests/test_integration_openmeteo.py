import datetime as dt

import pytest

from openenergy.providers.base import GeoPoint
from openenergy.providers.openmeteo.provider import OpenMeteoProvider


@pytest.mark.integration
def test_live_forecast_returns_data():
    prov = OpenMeteoProvider()
    point = GeoPoint(point_id=1, latitude=39.93, longitude=32.86)
    series = prov.fetch_forecast(point, ["wind_speed_100m", "temperature_2m"], forecast_days=2)
    assert series.frame.height > 0
    assert "wind_speed_100m" in set(series.frame["variable"].unique())
