import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
import respx

from openenergy.providers.base import ApiRole, GeoPoint
from openenergy.providers.openmeteo.client import (
    BASE_URLS,
    CUSTOMER_URLS,
    OpenMeteoClient,
)
from openenergy.providers.openmeteo.provider import (
    OpenMeteoProvider,
    SingleRunsDisabledError,
)

FIX = Path(__file__).parent.parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


@respx.mock
def test_fetch_forecast_parses_long_format():
    respx.get(BASE_URLS[ApiRole.FORECAST]).mock(
        return_value=httpx.Response(200, json=_load("forecast_hourly.json"))
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=1, latitude=39.9, longitude=32.8)
    series = prov.fetch_forecast(point, ["wind_speed_100m", "temperature_2m"])
    assert series.role == ApiRole.FORECAST.value
    f = series.frame
    assert set(f["variable"].unique()) == {"wind_speed_100m", "temperature_2m"}
    assert f.height == 4  # 2 zaman x 2 değişken
    row = f.filter((f["variable"] == "wind_speed_100m")).sort("valid_time").row(0, named=True)
    assert row["value"] == 7.2
    assert row["valid_time"] == dt.datetime(2024, 6, 1, 0, 0)


@respx.mock
def test_fetch_archive_uses_archive_host():
    route = respx.get(BASE_URLS[ApiRole.ARCHIVE]).mock(
        return_value=httpx.Response(200, json=_load("archive_hourly.json"))
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=2, latitude=39.9, longitude=32.8)
    series = prov.fetch_archive(point, ["wind_speed_100m"],
                               start_date=dt.date(2023, 1, 1), end_date=dt.date(2023, 1, 1))
    assert route.called
    assert series.role == ApiRole.ARCHIVE.value
    assert series.frame.height == 2


@respx.mock
def test_fetch_previous_runs_builds_lead_buckets():
    respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=_load("previous_runs.json"))
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=3, latitude=39.9, longitude=32.8)
    series = prov.fetch_previous_runs(point, ["wind_speed_100m"], previous_days=2)
    f = series.frame
    assert set(f["lead_hours"].unique()) == {0, 24, 48}
    # day1 bucket: valid 00:00 → issue 24h önce, value 8.5
    r = f.filter((f["lead_hours"] == 24)).sort("valid_time").row(0, named=True)
    assert r["value"] == 8.5
    assert r["issue_time"] == dt.datetime(2024, 6, 1, 0, 0)
    # tüm variable adları suffix'siz "wind_speed_100m"
    assert set(f["variable"].unique()) == {"wind_speed_100m"}


@respx.mock
def test_fetch_previous_runs_expands_hourly_with_previous_day_suffixes():
    # Open-Meteo previous-runs API'sinde geçmiş run'lar AYRI bir param ile değil,
    # hourly'ye `<var>_previous_dayN` ekleri istenerek gelir. `previous_days` query
    # param OLARAK GÖNDERİLMEZ (API onu yok sayar).
    route = respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=_load("previous_runs.json"))
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=3, latitude=39.9, longitude=32.8)
    prov.fetch_previous_runs(point, ["wind_speed_100m"], previous_days=2)
    params = route.calls.last.request.url.params
    assert "previous_days" not in params
    hourly = params["hourly"].split(",")
    assert hourly == [
        "wind_speed_100m",
        "wind_speed_100m_previous_day1",
        "wind_speed_100m_previous_day2",
    ]


@respx.mock
def test_grid_fetch_batches_25_points_into_one_request():
    payloads = [
        {
            "latitude": 39 + i / 100,
            "longitude": 28,
            "hourly": {
                "time": ["2026-07-18T00:00"],
                "wind_speed_100m": [float(i)],
            },
        }
        for i in range(25)
    ]
    route = respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=payloads)
    )
    provider = OpenMeteoProvider(OpenMeteoClient())
    points = [
        GeoPoint(point_id=i + 1, latitude=39 + i / 100, longitude=28)
        for i in range(25)
    ]

    result = provider.fetch_previous_runs_batch(
        points, ["wind_speed_100m"], past_days=7, previous_days=0
    )

    assert route.call_count == 1
    assert [series.point_id for series in result] == list(range(1, 26))
    assert route.calls.last.request.url.params["latitude"].count(",") == 24
    assert [series.frame["value"][0] for series in result] == [
        float(i) for i in range(25)
    ]


@respx.mock
def test_fetch_previous_runs_strips_model_suffix_from_column():
    payload = {
        "latitude": 39.9, "longitude": 32.8, "timezone": "GMT",
        "hourly": {
            "time": ["2024-06-02T00:00", "2024-06-02T01:00"],
            "wind_speed_100m_era5_previous_day1": [8.5, 8.9],
            "wind_speed_100m_era5": [9.0, 9.4],
        },
    }
    respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=payload)
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=5, latitude=39.9, longitude=32.8)
    series = prov.fetch_previous_runs(point, ["wind_speed_100m"], models=["era5"])
    f = series.frame
    # Both columns should yield variable "wind_speed_100m", stripping "era5" suffix
    assert set(f["variable"].unique()) == {"wind_speed_100m"}
    # _previous_day1 column → lead_hours=24; non-previous column → lead_hours=0
    assert set(f["lead_hours"].unique()) == {0, 24}


@respx.mock
def test_fetch_previous_runs_day5_yields_lead_120():
    # F1-t1: previous_day5 columns are day-5 reforecasts → lead_hours = 5*24 = 120,
    # the leakage-safe training weather for a 120h (day-6) horizon.
    payload = {
        "latitude": 39.9, "longitude": 32.8, "timezone": "GMT",
        "hourly": {
            "time": ["2024-06-06T00:00", "2024-06-06T01:00"],
            "wind_speed_100m": [9.0, 9.4],
            "wind_speed_100m_previous_day5": [8.5, 8.9],
        },
    }
    respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=payload)
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=6, latitude=39.9, longitude=32.8)
    series = prov.fetch_previous_runs(point, ["wind_speed_100m"], previous_days=5)
    f = series.frame
    assert 120 in set(f["lead_hours"].unique())
    r = f.filter(f["lead_hours"] == 120).sort("valid_time").row(0, named=True)
    assert r["value"] == 8.5
    # lead 120 → issued 120h before valid_time
    assert r["issue_time"] == dt.datetime(2024, 6, 1, 0, 0)


def test_fetch_single_run_disabled_raises_clear_error():
    # Gated OFF (default) → clear error, no network call.
    prov = OpenMeteoProvider(OpenMeteoClient(api_key="secret"))
    point = GeoPoint(point_id=7, latitude=39.9, longitude=32.8)
    with pytest.raises(SingleRunsDisabledError):
        prov.fetch_single_run(point, ["wind_speed_100m"],
                              run=dt.datetime(2024, 6, 1))


def test_fetch_single_run_enabled_requires_api_key():
    # Enabled but no API key (paid endpoint) → clear error, no network call.
    prov = OpenMeteoProvider(OpenMeteoClient(), single_runs_enabled=True)
    point = GeoPoint(point_id=7, latitude=39.9, longitude=32.8)
    with pytest.raises(SingleRunsDisabledError):
        prov.fetch_single_run(point, ["wind_speed_100m"],
                              run=dt.datetime(2024, 6, 1))


@respx.mock
def test_fetch_single_run_parses_issue_and_lead_provenance():
    # A single run: hourly.time are valid_times of ONE initialization; issue_time
    # is the run, lead_hours = valid_time - run (arbitrary hourly leads).
    run = dt.datetime(2024, 6, 1, 0, 0)
    payload = {
        "latitude": 39.9, "longitude": 32.8, "timezone": "GMT",
        "hourly": {
            "time": ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T05:00"],
            "wind_speed_100m": [7.0, 7.5, 9.0],
        },
    }
    route = respx.get(CUSTOMER_URLS[ApiRole.SINGLE_RUNS]).mock(
        return_value=httpx.Response(200, json=payload)
    )
    prov = OpenMeteoProvider(OpenMeteoClient(api_key="secret"),
                             single_runs_enabled=True)
    point = GeoPoint(point_id=8, latitude=39.9, longitude=32.8)
    series = prov.fetch_single_run(point, ["wind_speed_100m"], run=run)

    assert route.called
    # run sent as query param (unix seconds)
    assert "run" in route.calls.last.request.url.params

    assert series.role == ApiRole.SINGLE_RUNS.value
    f = series.frame
    assert set(f["variable"].unique()) == {"wind_speed_100m"}
    # arbitrary hourly leads, NOT restricted to multiples of 24
    assert set(f["lead_hours"].unique()) == {0, 1, 5}
    for r in f.iter_rows(named=True):
        assert r["issue_time"] == run
    r5 = f.filter(f["lead_hours"] == 5).row(0, named=True)
    assert r5["value"] == 9.0
    assert r5["valid_time"] == dt.datetime(2024, 6, 1, 5, 0)


@respx.mock
def test_fetch_single_run_strips_model_suffix():
    run = dt.datetime(2024, 6, 1, 0, 0)
    payload = {
        "latitude": 39.9, "longitude": 32.8, "timezone": "GMT",
        "hourly": {
            "time": ["2024-06-01T00:00", "2024-06-01T01:00"],
            "wind_speed_100m_icon_eu": [7.0, 7.5],
        },
    }
    respx.get(CUSTOMER_URLS[ApiRole.SINGLE_RUNS]).mock(
        return_value=httpx.Response(200, json=payload)
    )
    prov = OpenMeteoProvider(OpenMeteoClient(api_key="secret"),
                             single_runs_enabled=True)
    point = GeoPoint(point_id=9, latitude=39.9, longitude=32.8)
    series = prov.fetch_single_run(point, ["wind_speed_100m"],
                                   run=run, models=["icon_eu"])
    f = series.frame
    assert set(f["variable"].unique()) == {"wind_speed_100m"}
    assert series.model == "icon_eu"


@respx.mock
def test_fetch_ensemble_preserves_members():
    respx.get(BASE_URLS[ApiRole.ENSEMBLE]).mock(
        return_value=httpx.Response(200, json=_load("ensemble.json"))
    )
    prov = OpenMeteoProvider(OpenMeteoClient())
    point = GeoPoint(point_id=4, latitude=39.9, longitude=32.8)
    series = prov.fetch_ensemble(point, ["wind_speed_100m"], models=["icon_seamless"])
    f = series.frame
    variables = set(f["variable"].unique())
    assert "wind_speed_100m_member00" in variables
    assert "wind_speed_100m_member01" in variables
    assert "wind_speed_100m_member02" in variables
