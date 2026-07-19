import polars as pl
import pytest

from openenergy.providers.base import (
    LONG_COLUMNS,
    ApiRole,
    GeoPoint,
    WeatherProvider,
    WeatherSeries,
)
from openenergy.providers.openmeteo import variables as V


def test_api_roles():
    assert ApiRole.PREVIOUS_RUNS.value == "previous_runs"
    assert ApiRole.SINGLE_RUNS.value == "single_runs"
    assert {r.value for r in ApiRole} == {
        "forecast", "archive", "previous_runs", "ensemble", "single_runs"}


def test_weather_series_empty_has_long_schema():
    s = WeatherSeries.empty(point_id=1, role=ApiRole.FORECAST.value, model="icon")
    assert s.frame.columns == LONG_COLUMNS
    assert s.frame.height == 0


def test_geopoint_optional_fields_default_none():
    g = GeoPoint(point_id=1, latitude=39.9, longitude=32.8)
    assert g.hub_height_m is None and g.tilt is None


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        WeatherProvider()  # type: ignore[abstract]


def test_default_variables_by_kind():
    assert "wind_speed_100m" in V.default_variables("wind")
    assert "global_tilted_irradiance" in V.default_variables("solar")
    assert "temperature_2m" in V.default_variables("wind")
