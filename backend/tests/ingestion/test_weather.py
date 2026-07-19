import datetime as dt

import httpx
import polars as pl
import pytest
import respx

from openenergy.assets.repository import PlantPoint
from openenergy.ingestion.weather import (
    ingest_grid_previous_runs,
    ingest_previous_runs,
    ingest_serving_previous_runs,
    ingest_single_run,
)
from openenergy.ingestion.weather_writer import read_weather
from openenergy.providers.base import ApiRole, GeoPoint, WeatherProvider, WeatherSeries
from openenergy.providers.openmeteo.client import BASE_URLS, OpenMeteoClient
from openenergy.providers.openmeteo.provider import OpenMeteoProvider
from openenergy.providers.openmeteo.provider import SingleRunsDisabledError
from openenergy.storage import connect, init_schema


class FakeProvider(WeatherProvider):
    """previous-runs çağrısına lead 0/24/48 içeren küçük bir seri döndürür."""

    name = "fake"

    def fetch_previous_runs(self, point: GeoPoint, variables, **kwargs) -> WeatherSeries:
        base = dt.datetime(2026, 1, 1)
        rows = []
        for lead in (0, 24, 48):
            for h in range(5):
                vt = base + dt.timedelta(hours=h)
                rows.append({
                    "valid_time": vt,
                    "issue_time": vt - dt.timedelta(hours=lead),
                    "lead_hours": lead,
                    "variable": "wind_speed_100m",
                    "value": 5.0 + h,
                })
        frame = pl.DataFrame(rows, schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        })
        return WeatherSeries(point_id=point.point_id, role=ApiRole.PREVIOUS_RUNS.value,
                             model="fake", frame=frame)

    def fetch_forecast(self, *a, **k): ...
    def fetch_archive(self, *a, **k): ...
    def fetch_ensemble(self, *a, **k): ...


class ModelEchoProvider(WeatherProvider):
    """Echoes the requested model so multi-source storage can be verified."""

    name = "echo"

    def fetch_previous_runs(self, point: GeoPoint, variables, *, models=None, **kwargs) -> WeatherSeries:
        model = models[0] if models else "best_match"
        base = dt.datetime(2026, 1, 1)
        frame = pl.DataFrame(
            {
                "valid_time": [base], "issue_time": [base - dt.timedelta(hours=24)],
                "lead_hours": [24], "variable": ["wind_speed_100m"], "value": [5.0],
            },
            schema_overrides={
                "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                "lead_hours": pl.Int32, "value": pl.Float64,
            },
        )
        return WeatherSeries(point_id=point.point_id, role=ApiRole.PREVIOUS_RUNS.value,
                             model=model, frame=frame)

    def fetch_forecast(self, *a, **k): ...
    def fetch_archive(self, *a, **k): ...
    def fetch_ensemble(self, *a, **k): ...


def test_ingest_previous_runs_stores_source_model():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=3, plant_id="wf1", point_type="weather", latitude=39, longitude=28)
    ingest_previous_runs(con, point, "wind", provider=ModelEchoProvider(), models=["icon_eu"])
    m = con.execute("SELECT DISTINCT model FROM weather_raw WHERE point_id=3").fetchall()
    assert m == [("icon_eu",)]


def test_ingest_grid_loops_points_and_tolerates_failure():
    con = connect(None)
    init_schema(con)
    pts = [PlantPoint(point_id=i, plant_id="wf1", point_type="weather",
                      latitude=39 + i * 0.1, longitude=28) for i in (10, 11, 12)]

    class FlakyProvider(ModelEchoProvider):
        def fetch_previous_runs(self, point, variables, **kwargs):
            if point.point_id == 11:
                raise RuntimeError("boom")
            return super().fetch_previous_runs(point, variables, **kwargs)

    res = ingest_grid_previous_runs(con, pts, "wind", provider=FlakyProvider())
    assert res["points"] == 3
    assert res["ingested"] == 2  # point 11 failed but batch continued
    assert len(res["errors"]) == 1
    assert res["rows"] == 2


def test_delta_ingest_uses_watermark_derived_past_days():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(
        point_id=13,
        plant_id="wf1",
        point_type="weather",
        latitude=39,
        longitude=28,
    )
    watermark = (
        dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0)
        - dt.timedelta(days=3)
    )
    for lead in (0, 24):
        con.execute(
            """
            INSERT INTO weather_raw
              (point_id, role, model, valid_time, issue_time, lead_hours, variable, value)
            VALUES (?, 'previous_runs', 'icon_eu', ?, ?, ?, 'wind_speed_100m', 5.0)
            """,
            [point.point_id, watermark, watermark - dt.timedelta(hours=lead), lead],
        )
    calls: list[dict] = []

    class RecordingProvider(ModelEchoProvider):
        def fetch_previous_runs(self, point, variables, **kwargs):
            calls.append(kwargs)
            return super().fetch_previous_runs(point, variables, **kwargs)

    result = ingest_previous_runs(
        con,
        point,
        "wind",
        past_days=92,
        previous_days=1,
        models=["icon_eu"],
        provider=RecordingProvider(),
    )

    assert calls[0]["past_days"] == 3
    assert result["past_days"] == 3


@respx.mock
def test_grid_ingest_batches_25_points_into_one_request():
    con = connect(None)
    init_schema(con)
    points = [
        PlantPoint(
            point_id=i + 100,
            plant_id="wf1",
            point_type="weather",
            latitude=39 + i / 100,
            longitude=28,
        )
        for i in range(25)
    ]
    payloads = [
        {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "hourly": {
                "time": ["2026-07-18T00:00"],
                "wind_speed_100m": [5.0],
            },
        }
        for point in points
    ]
    route = respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=payloads)
    )

    result = ingest_grid_previous_runs(
        con,
        points,
        "wind",
        sources=["icon_eu"],
        past_days=7,
        previous_days=0,
        provider=OpenMeteoProvider(OpenMeteoClient()),
    )

    assert route.call_count == 1
    assert result == {"points": 25, "ingested": 25, "rows": 25, "errors": []}


class SingleRunProvider(WeatherProvider):
    """Returns a stubbed single-run series (arbitrary hourly leads)."""

    name = "singlerun"

    def fetch_single_run(self, point: GeoPoint, variables, *, run,
                         forecast_days=7, models=None) -> WeatherSeries:
        rows = []
        for lead in (0, 1, 5):
            vt = run + dt.timedelta(hours=lead)
            rows.append({
                "valid_time": vt,
                "issue_time": run,
                "lead_hours": lead,
                "variable": "wind_speed_100m",
                "value": 5.0 + lead,
            })
        frame = pl.DataFrame(rows, schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        })
        return WeatherSeries(point_id=point.point_id,
                             role=ApiRole.SINGLE_RUNS.value,
                             model="fake", frame=frame)

    def fetch_forecast(self, *a, **k): ...
    def fetch_archive(self, *a, **k): ...
    def fetch_previous_runs(self, *a, **k): ...
    def fetch_ensemble(self, *a, **k): ...


def test_ingest_single_run_stores_provenance():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=20, plant_id="wf1", point_type="turbine",
                       latitude=39.6, longitude=27.0, hub_height_m=100)
    run = dt.datetime(2026, 1, 1, 0, 0)

    res = ingest_single_run(con, point, "wind", run=run,
                            provider=SingleRunProvider())

    assert res["rows"] == 3
    assert res["lead_hours"] == [0, 1, 5]

    stored = read_weather(con, 20, ApiRole.SINGLE_RUNS.value)
    assert stored.height == 3
    assert set(stored["lead_hours"].unique().to_list()) == {0, 1, 5}
    assert set(stored["issue_time"].unique().to_list()) == {run}


def test_ingest_single_run_gated_off_raises():
    # Default OpenMeteoProvider path is gated OFF → clear error, nothing stored.
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=21, plant_id="wf1", point_type="turbine",
                       latitude=39.6, longitude=27.0, hub_height_m=100)
    with pytest.raises(SingleRunsDisabledError):
        ingest_single_run(con, point, "wind", run=dt.datetime(2026, 1, 1))
    assert con.execute(
        "SELECT count(*) FROM weather_raw WHERE point_id=21").fetchone()[0] == 0


def test_ingest_previous_runs_writes_weather():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=1, plant_id="wf1", point_type="turbine",
                       latitude=39.6, longitude=27.0, hub_height_m=100)

    res = ingest_previous_runs(con, point, "wind", provider=FakeProvider())

    assert res["rows"] == 15
    assert res["lead_hours"] == [0, 24, 48]

    stored = read_weather(con, 1, ApiRole.PREVIOUS_RUNS.value)
    assert stored.height == 15
    assert set(stored["lead_hours"].unique().to_list()) == {0, 24, 48}


def test_ingest_serving_previous_runs_bounds_one_call_to_requested_horizons():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=22, plant_id="wf1", point_type="turbine",
                       latitude=39.6, longitude=27.0, hub_height_m=100)
    calls = []

    class RecordingProvider(FakeProvider):
        def fetch_previous_runs(self, point, variables, **kwargs):
            calls.append(kwargs)
            return super().fetch_previous_runs(point, variables, **kwargs)

    result = ingest_serving_previous_runs(
        con, point, "wind", horizon_hours_list=[48, 24, 48],
        models=["ecmwf"], provider=RecordingProvider(),
    )

    assert len(calls) == 1
    assert calls[0]["past_days"] == 0
    assert calls[0]["previous_days"] == 2
    assert calls[0]["forecast_days"] == 3
    assert calls[0]["models"] == ["ecmwf"]
    assert result["requested_horizons"] == [24, 48]


def test_ingest_serving_previous_runs_rejects_non_daily_leads():
    con = connect(None)
    init_schema(con)
    point = PlantPoint(point_id=23, plant_id="wf1", point_type="turbine",
                       latitude=39.6, longitude=27.0, hub_height_m=100)
    with pytest.raises(ValueError, match="24"):
        ingest_serving_previous_runs(
            con, point, "wind", horizon_hours_list=[6], provider=FakeProvider(),
        )
