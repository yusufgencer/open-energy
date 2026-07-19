from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum

import polars as pl
from pydantic import BaseModel

LONG_COLUMNS = ["valid_time", "issue_time", "lead_hours", "variable", "value"]

_LONG_SCHEMA = {
    "valid_time": pl.Datetime("us"),
    "issue_time": pl.Datetime("us"),
    "lead_hours": pl.Int32,
    "variable": pl.Utf8,
    "value": pl.Float64,
}


class ApiRole(str, Enum):
    FORECAST = "forecast"
    ARCHIVE = "archive"
    PREVIOUS_RUNS = "previous_runs"
    ENSEMBLE = "ensemble"
    # Single Runs API: one pinned model initialization (run=) yielding arbitrary
    # leak-free leads. Served by the previous-runs host; paid/config-gated.
    SINGLE_RUNS = "single_runs"


class GeoPoint(BaseModel):
    point_id: int
    latitude: float
    longitude: float
    hub_height_m: float | None = None
    tilt: float | None = None
    azimuth: float | None = None


class WeatherSeries(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    point_id: int
    role: str
    model: str
    frame: pl.DataFrame

    @classmethod
    def empty(cls, point_id: int, role: str, model: str) -> "WeatherSeries":
        return cls(point_id=point_id, role=role, model=model,
                   frame=pl.DataFrame(schema=_LONG_SCHEMA))


class WeatherProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_forecast(self, point: GeoPoint, variables: list[str], *,
                       forecast_days: int = 7, models: list[str] | None = None) -> WeatherSeries: ...

    @abstractmethod
    def fetch_archive(self, point: GeoPoint, variables: list[str], *,
                      start_date: date, end_date: date,
                      models: list[str] | None = None) -> WeatherSeries: ...

    @abstractmethod
    def fetch_previous_runs(self, point: GeoPoint, variables: list[str], *,
                            past_days: int = 60, forecast_days: int = 1,
                            previous_days: int = 3,
                            models: list[str] | None = None) -> WeatherSeries: ...

    @abstractmethod
    def fetch_ensemble(self, point: GeoPoint, variables: list[str], *,
                       forecast_days: int = 7, models: list[str] | None = None) -> WeatherSeries: ...

    def fetch_single_run(self, point: GeoPoint, variables: list[str], *,
                         run: datetime, forecast_days: int = 7,
                         models: list[str] | None = None) -> WeatherSeries:
        """Single Runs API (paid, config-gated). Not supported by default."""
        raise NotImplementedError(
            f"{self.name} provider does not support the Single Runs API")

    def fetch_previous_runs_batch(
        self,
        points: list[GeoPoint],
        variables: list[str],
        *,
        past_days: int = 60,
        forecast_days: int = 1,
        previous_days: int = 3,
        models: list[str] | None = None,
    ) -> list[WeatherSeries]:
        """Compatibility fallback for providers without multi-location support."""
        return [
            self.fetch_previous_runs(
                point,
                variables,
                past_days=past_days,
                forecast_days=forecast_days,
                previous_days=previous_days,
                models=models,
            )
            for point in points
        ]
