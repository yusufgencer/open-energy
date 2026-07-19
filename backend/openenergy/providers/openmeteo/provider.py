from __future__ import annotations

import datetime as dt
import re

import polars as pl

from openenergy.providers.base import (
    ApiRole,
    GeoPoint,
    WeatherProvider,
    WeatherSeries,
)
from openenergy.providers.openmeteo.client import OpenMeteoClient

_KNOWN_MODEL_SUFFIXES = (
    "icon_seamless", "icon_global", "icon_eu", "icon_d2",
    "gfs_seamless", "gfs_global", "ecmwf_ifs025", "ecmwf_aifs025",
    "meteofrance_seamless", "gem_seamless", "era5", "era5_land",
)

_PREV_RE = re.compile(r"^(?P<var>.+)_previous_day(?P<n>\d+)$")
_MEMBER_RE = re.compile(r"^(?P<var>.+)_member(?P<n>\d+)$")


class SingleRunsDisabledError(RuntimeError):
    """Raised when the paid, config-gated Single Runs API is used while OFF."""


def _split_var_model(column: str, default_model: str) -> tuple[str, str]:
    for suffix in _KNOWN_MODEL_SUFFIXES:
        if column.endswith("_" + suffix):
            return column[: -(len(suffix) + 1)], suffix
    return column, default_model


def _parse_hourly(payload: dict, point_id: int, role: ApiRole,
                  default_model: str) -> WeatherSeries:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherSeries.empty(point_id, role.value, default_model)
    valid = [dt.datetime.fromisoformat(t) for t in times]
    blocks: list[pl.DataFrame] = []
    model_seen = default_model
    for column, values in hourly.items():
        if column == "time":
            continue
        variable, model = _split_var_model(column, default_model)
        model_seen = model
        blocks.append(
            pl.DataFrame(
                {
                    "valid_time": valid,
                    "issue_time": [None] * len(valid),
                    "lead_hours": [None] * len(valid),
                    "variable": [variable] * len(valid),
                    "value": [float(v) if v is not None else None for v in values],
                },
                schema_overrides={
                    "valid_time": pl.Datetime("us"),
                    "issue_time": pl.Datetime("us"),
                    "lead_hours": pl.Int32,
                    "value": pl.Float64,
                },
            )
        )
    frame = pl.concat(blocks) if blocks else pl.DataFrame()
    return WeatherSeries(point_id=point_id, role=role.value, model=model_seen, frame=frame)


def _parse_previous_runs(payload: dict, point_id: int, default_model: str) -> WeatherSeries:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherSeries.empty(point_id, ApiRole.PREVIOUS_RUNS.value, default_model)
    valid = [dt.datetime.fromisoformat(t) for t in times]
    blocks: list[pl.DataFrame] = []
    for column, values in hourly.items():
        if column == "time":
            continue
        m = _PREV_RE.match(column)
        if m:
            variable, _ = _split_var_model(m.group("var"), default_model)
            n = int(m.group("n"))
        else:
            variable, _ = _split_var_model(column, default_model)
            n = 0
        lead = n * 24
        issue = [v - dt.timedelta(hours=lead) for v in valid]
        blocks.append(
            pl.DataFrame(
                {
                    "valid_time": valid,
                    "issue_time": issue,
                    "lead_hours": [lead] * len(valid),
                    "variable": [variable] * len(valid),
                    "value": [float(v) if v is not None else None for v in values],
                },
                schema_overrides={
                    "valid_time": pl.Datetime("us"),
                    "issue_time": pl.Datetime("us"),
                    "lead_hours": pl.Int32,
                    "value": pl.Float64,
                },
            )
        )
    frame = pl.concat(blocks) if blocks else pl.DataFrame()
    return WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                         model=default_model, frame=frame)


def _parse_single_run(payload: dict, point_id: int, run: dt.datetime,
                      default_model: str) -> WeatherSeries:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherSeries.empty(point_id, ApiRole.SINGLE_RUNS.value, default_model)
    valid = [dt.datetime.fromisoformat(t) for t in times]
    leads = [int((v - run).total_seconds() // 3600) for v in valid]
    blocks: list[pl.DataFrame] = []
    model_seen = default_model
    for column, values in hourly.items():
        if column == "time":
            continue
        variable, model = _split_var_model(column, default_model)
        model_seen = model
        blocks.append(
            pl.DataFrame(
                {
                    "valid_time": valid,
                    "issue_time": [run] * len(valid),
                    "lead_hours": leads,
                    "variable": [variable] * len(valid),
                    "value": [float(v) if v is not None else None for v in values],
                },
                schema_overrides={
                    "valid_time": pl.Datetime("us"),
                    "issue_time": pl.Datetime("us"),
                    "lead_hours": pl.Int32,
                    "value": pl.Float64,
                },
            )
        )
    frame = pl.concat(blocks) if blocks else pl.DataFrame()
    return WeatherSeries(point_id=point_id, role=ApiRole.SINGLE_RUNS.value,
                         model=model_seen, frame=frame)


def _parse_ensemble(payload: dict, point_id: int, default_model: str) -> WeatherSeries:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherSeries.empty(point_id, ApiRole.ENSEMBLE.value, default_model)
    valid = [dt.datetime.fromisoformat(t) for t in times]
    blocks: list[pl.DataFrame] = []
    for column, values in hourly.items():
        if column == "time":
            continue
        if _MEMBER_RE.match(column):
            variable = column
        else:
            variable = f"{column}_member00"
        blocks.append(
            pl.DataFrame(
                {
                    "valid_time": valid,
                    "issue_time": [None] * len(valid),
                    "lead_hours": [None] * len(valid),
                    "variable": [variable] * len(valid),
                    "value": [float(v) if v is not None else None for v in values],
                },
                schema_overrides={
                    "valid_time": pl.Datetime("us"),
                    "issue_time": pl.Datetime("us"),
                    "lead_hours": pl.Int32,
                    "value": pl.Float64,
                },
            )
        )
    frame = pl.concat(blocks) if blocks else pl.DataFrame()
    return WeatherSeries(point_id=point_id, role=ApiRole.ENSEMBLE.value,
                         model=default_model, frame=frame)


class OpenMeteoProvider(WeatherProvider):
    name = "openmeteo"

    def __init__(self, client: OpenMeteoClient | None = None, *,
                 single_runs_enabled: bool = False) -> None:
        self.client = client or OpenMeteoClient()
        # Paid Single Runs API is OFF unless BOTH this flag and an API key are set.
        self.single_runs_enabled = single_runs_enabled

    def _base_params(self, point: GeoPoint, variables: list[str]) -> dict:
        params: dict = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "hourly": ",".join(variables),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        if point.tilt is not None:
            params["tilt"] = point.tilt
        if point.azimuth is not None:
            params["azimuth"] = point.azimuth
        return params

    def _base_params_many(self, points: list[GeoPoint], variables: list[str]) -> dict:
        if not points:
            raise ValueError("points cannot be empty")
        params: dict = {
            "latitude": ",".join(str(point.latitude) for point in points),
            "longitude": ",".join(str(point.longitude) for point in points),
            "hourly": ",".join(variables),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        # Open-Meteo accepts comma-separated per-location solar orientation.
        # Omit an optional field unless every location supplies it.
        if all(point.tilt is not None for point in points):
            params["tilt"] = ",".join(str(point.tilt) for point in points)
        if all(point.azimuth is not None for point in points):
            params["azimuth"] = ",".join(str(point.azimuth) for point in points)
        return params

    def fetch_forecast(self, point: GeoPoint, variables: list[str], *,
                       forecast_days: int = 7, models: list[str] | None = None) -> WeatherSeries:
        params = self._base_params(point, variables)
        params["forecast_days"] = forecast_days
        if models:
            params["models"] = ",".join(models)
        payload = self.client.get(ApiRole.FORECAST, params)
        default_model = models[0] if models else "best_match"
        return _parse_hourly(payload, point.point_id, ApiRole.FORECAST, default_model)

    def fetch_archive(self, point: GeoPoint, variables: list[str], *,
                      start_date: dt.date, end_date: dt.date,
                      models: list[str] | None = None) -> WeatherSeries:
        params = self._base_params(point, variables)
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()
        params["models"] = ",".join(models) if models else "era5"
        payload = self.client.get(ApiRole.ARCHIVE, params)
        default_model = models[0] if models else "era5"
        return _parse_hourly(payload, point.point_id, ApiRole.ARCHIVE, default_model)

    def fetch_previous_runs(self, point: GeoPoint, variables: list[str], *,
                            past_days: int = 60, forecast_days: int = 1,
                            previous_days: int = 3,
                            models: list[str] | None = None) -> WeatherSeries:
        # Open-Meteo previous-runs API'sinde geçmiş run'lar ayrı bir param ile DEĞİL,
        # her değişkene `_previous_dayN` eki istenerek gelir. Base değişken = lead 0,
        # `_previous_day1` = lead 24, `_previous_day2` = lead 48 ... (bkz. _parse_previous_runs).
        expanded: list[str] = []
        for var in variables:
            expanded.append(var)
            for n in range(1, previous_days + 1):
                expanded.append(f"{var}_previous_day{n}")
        params = self._base_params(point, expanded)
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days
        if models:
            params["models"] = ",".join(models)
        payload = self.client.get(ApiRole.PREVIOUS_RUNS, params)
        default_model = models[0] if models else "best_match"
        return _parse_previous_runs(payload, point.point_id, default_model)

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
        """Fetch and split Open-Meteo's multi-location response in one request."""
        if not points:
            return []
        expanded: list[str] = []
        for var in variables:
            expanded.append(var)
            expanded.extend(f"{var}_previous_day{n}" for n in range(1, previous_days + 1))
        params = self._base_params_many(points, expanded)
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days
        if models:
            params["models"] = ",".join(models)
        payload = self.client.get(ApiRole.PREVIOUS_RUNS, params, weight=len(points))
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(points):
            raise ValueError(
                "Open-Meteo multi-location response count does not match request "
                f"({len(payloads)} != {len(points)})"
            )
        default_model = models[0] if models else "best_match"
        return [
            _parse_previous_runs(item, point.point_id, default_model)
            for point, item in zip(points, payloads, strict=True)
        ]

    def fetch_ensemble(self, point: GeoPoint, variables: list[str], *,
                       forecast_days: int = 7, models: list[str] | None = None) -> WeatherSeries:
        params = self._base_params(point, variables)
        params["forecast_days"] = forecast_days
        params["models"] = ",".join(models) if models else "icon_seamless"
        payload = self.client.get(ApiRole.ENSEMBLE, params)
        default_model = models[0] if models else "icon_seamless"
        return _parse_ensemble(payload, point.point_id, default_model)

    def fetch_single_run(self, point: GeoPoint, variables: list[str], *,
                         run: dt.datetime, forecast_days: int = 7,
                         models: list[str] | None = None) -> WeatherSeries:
        """Single Runs API — one pinned model run (`run=`) → arbitrary leak-free leads.

        Paid endpoint, gated OFF by default: requires both `single_runs_enabled`
        AND an API key on the client. issue_time = run; lead_hours = valid_time-run.
        """
        if not self.single_runs_enabled:
            raise SingleRunsDisabledError(
                "Open-Meteo Single Runs API is disabled; set "
                "OPENENERGY_OPENMETEO_SINGLE_RUNS_ENABLED=1 to enable it")
        if not self.client.api_key:
            raise SingleRunsDisabledError(
                "Open-Meteo Single Runs API is paid; an API key is required "
                "(set OPENENERGY_OPENMETEO_API_KEY)")
        params = self._base_params(point, variables)
        params["forecast_days"] = forecast_days
        # run= selects one model initialization by Unix timestamp (treat naive as UTC).
        run_utc = run if run.tzinfo else run.replace(tzinfo=dt.timezone.utc)
        params["run"] = int(run_utc.timestamp())
        if models:
            params["models"] = ",".join(models)
        payload = self.client.get(ApiRole.SINGLE_RUNS, params)
        default_model = models[0] if models else "best_match"
        return _parse_single_run(payload, point.point_id, run, default_model)
