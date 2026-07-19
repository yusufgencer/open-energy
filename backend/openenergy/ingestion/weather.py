"""Hava verisi ingestion — leakage-safe eğitim için Open-Meteo previous-runs.

Eğitim veri seti `weather_raw` tablosundan role=previous_runs ve lead_hours=horizon
ile okunur (bkz. datasets/horizon.py). Open-Meteo previous-runs API'si lead'i tam gün
katlarında verir (lead = N*24), dolayısıyla yalnızca 24'ün katı horizon'lar
(24, 48, ...) eşleşir. `previous_days` kaç günlük geçmiş run getirileceğini belirler:
previous_days=2 → lead 0/24/48; previous_days=7 → lead 0/24/.../168 (F1-t1: gün
3–7 horizonları 72/96/120/144/168 için leakage-safe eğitim verisi).
"""
from __future__ import annotations

import datetime as dt
import math

import duckdb

from openenergy.assets.repository import PlantPoint
from openenergy.config import get_settings
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.providers.base import GeoPoint, WeatherProvider
from openenergy.providers.openmeteo.client import OpenMeteoClient
from openenergy.providers.openmeteo.provider import OpenMeteoProvider
from openenergy.providers.openmeteo.variables import default_variables


def _to_geopoint(point: PlantPoint) -> GeoPoint:
    if point.point_id is None:
        raise ValueError("point_id gerekli")
    return GeoPoint(
        point_id=point.point_id,
        latitude=point.latitude,
        longitude=point.longitude,
        hub_height_m=point.hub_height_m,
        tilt=point.tilt,
        azimuth=point.azimuth,
    )


def _watermark_past_days(
    con: duckdb.DuckDBPyConnection,
    *,
    point_id: int,
    source: str | None,
    role: str,
    lead_hours: list[int],
    requested_past_days: int,
    as_of: dt.datetime | None = None,
) -> int:
    """Shrink a historical request to the oldest complete high-water mark.

    Every requested lead must have a watermark for the same point/source/role;
    otherwise this is a bootstrap and the caller's full range is retained.
    """
    if requested_past_days <= 0 or not lead_hours:
        return max(0, requested_past_days)
    model = source or "best_match"
    placeholders = ",".join("?" for _ in lead_hours)
    rows = con.execute(
        f"""
        SELECT lead_hours, max(valid_time)
        FROM weather_raw
        WHERE point_id=? AND role=? AND model=?
          AND lead_hours IN ({placeholders})
        GROUP BY lead_hours
        """,
        [point_id, role, model, *lead_hours],
    ).fetchall()
    watermarks = {int(lead): value for lead, value in rows if value is not None}
    if any(lead not in watermarks for lead in lead_hours):
        return requested_past_days
    oldest = min(watermarks.values())
    current = as_of or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is not None:
        current = current.replace(tzinfo=None)
    # past_days is calendar-day based. Retaining the watermark day makes the
    # delta overlap idempotent and fills a partially ingested final day.
    delta = max(0, (current.date() - oldest.date()).days)
    return min(requested_past_days, delta)


def _series_result(
    con: duckdb.DuckDBPyConnection,
    series,
    *,
    past_days: int,
) -> dict:
    rows = write_weather_series(con, series)
    leads: list[int] = []
    n_vars = 0
    if series.frame.height > 0:
        leads = sorted({int(x) for x in series.frame["lead_hours"].unique().to_list()})
        n_vars = series.frame["variable"].n_unique()
    return {
        "rows": rows,
        "lead_hours": leads,
        "variables": n_vars,
        "past_days": past_days,
    }


def ingest_previous_runs(
    con: duckdb.DuckDBPyConnection,
    point: PlantPoint,
    kind: str,
    *,
    past_days: int = 92,
    previous_days: int = 7,
    forecast_days: int = 1,
    models: list[str] | None = None,
    provider: WeatherProvider | None = None,
) -> dict:
    """Leakage-safe eğitim hava verisini (previous-runs) çek ve weather_raw'a yaz.

    `models` verildiğinde tek bir NWP kaynağından çekilir ve weather_raw.model o
    kaynak olur (source_policy ekseninin veri kaynağı). None → provider varsayılanı.

    Returns: {"rows": int, "lead_hours": [...], "variables": int}
    """
    provider = provider or OpenMeteoProvider()
    variables = default_variables(kind)
    requested_leads = [day * 24 for day in range(previous_days + 1)]
    effective_past_days = _watermark_past_days(
        con,
        point_id=_to_geopoint(point).point_id,
        source=models[0] if models else None,
        role="previous_runs",
        lead_hours=requested_leads,
        requested_past_days=past_days,
    )
    series = provider.fetch_previous_runs(
        _to_geopoint(point),
        variables,
        past_days=effective_past_days,
        previous_days=previous_days,
        forecast_days=forecast_days,
        models=models,
    )
    return _series_result(con, series, past_days=effective_past_days)


def ingest_serving_previous_runs(
    con: duckdb.DuckDBPyConnection,
    point: PlantPoint,
    kind: str,
    *,
    horizon_hours_list: list[int],
    models: list[str] | None = None,
    provider: WeatherProvider | None = None,
) -> dict:
    """Bounded forward ingest for a future serving cycle.

    Open-Meteo previous-runs exposes only whole-day lead buckets. One call per
    point covers every requested serving horizon: request no historical tail,
    expand previous-day variables only through the largest lead, and request
    forward calendar days only through the farthest target (plus the current
    day). T-14's cycle may call this helper when its target rows are absent; this
    helper deliberately does not create a cycle or API endpoint.
    """
    horizons = sorted(set(horizon_hours_list))
    if not horizons:
        raise ValueError("horizon_hours_list boş olamaz")
    invalid = [h for h in horizons if h <= 0 or h % 24 != 0]
    if invalid:
        raise ValueError(
            "previous-runs serving yalnız pozitif 24 saat katlarını destekler: "
            f"{invalid}"
        )
    max_horizon = horizons[-1]
    result = ingest_previous_runs(
        con, point, kind,
        past_days=0,
        previous_days=max_horizon // 24,
        forecast_days=max(1, math.ceil(max_horizon / 24) + 1),
        models=models,
        provider=provider,
    )
    return {**result, "requested_horizons": horizons}


def ingest_single_run(
    con: duckdb.DuckDBPyConnection,
    point: PlantPoint,
    kind: str,
    *,
    run: dt.datetime,
    forecast_days: int = 7,
    models: list[str] | None = None,
    provider: WeatherProvider | None = None,
) -> dict:
    """Fetch one pinned Open-Meteo model run (Single Runs API) and store it.

    Paid, config-gated: when no provider is injected we build the default
    OpenMeteoProvider from settings — enabled ONLY if
    OPENENERGY_OPENMETEO_SINGLE_RUNS_ENABLED is set AND an API key is present.
    The provider raises SingleRunsDisabledError otherwise (no rows written).
    Stores role=single_runs with full (issue_time=run, lead_hours) provenance,
    giving arbitrary leak-free leads (not just multiples of 24h).

    Returns: {"rows": int, "lead_hours": [...], "variables": int}
    """
    if provider is None:
        settings = get_settings()
        provider = OpenMeteoProvider(
            OpenMeteoClient(api_key=settings.openmeteo_api_key),
            single_runs_enabled=settings.openmeteo_single_runs_enabled,
        )
    variables = default_variables(kind)
    series = provider.fetch_single_run(
        _to_geopoint(point),
        variables,
        run=run,
        forecast_days=forecast_days,
        models=models,
    )
    rows = write_weather_series(con, series)
    leads: list[int] = []
    n_vars = 0
    if series.frame.height > 0:
        leads = sorted({int(x) for x in series.frame["lead_hours"].unique().to_list()})
        n_vars = series.frame["variable"].n_unique()
    return {"rows": rows, "lead_hours": leads, "variables": n_vars}


def ingest_grid_previous_runs(
    con: duckdb.DuckDBPyConnection,
    points: list[PlantPoint],
    kind: str,
    *,
    sources: list[str] | None = None,
    past_days: int = 92,
    previous_days: int = 7,
    forecast_days: int = 1,
    provider: WeatherProvider | None = None,
) -> dict:
    """Ingest leakage-safe previous-runs weather for a whole grid of points.

    Loops every point × source, tolerating per-(point,source) failures so one bad
    cell never aborts the batch. `sources=None` uses the provider default; a list
    of NWP models feeds the source_policy axis. This is the data that backs every
    spatial candidate in the pool.

    Returns: {"points": int, "ingested": int, "rows": int, "errors": [...]}
    """
    if not points:
        return {"points": 0, "ingested": 0, "rows": 0, "errors": []}
    provider = provider or OpenMeteoProvider()
    source_list = sources if sources else [None]
    variables = default_variables(kind)
    requested_leads = [day * 24 for day in range(previous_days + 1)]
    total_rows = 0
    ingested = 0
    errors: list[dict] = []
    # Different watermarks require different date ranges. Points with the same
    # source/range remain batchable (a fresh 25-point grid is exactly one call).
    groups: dict[tuple[str | None, int], list[PlantPoint]] = {}
    for src in source_list:
        for point in points:
            if point.point_id is None:
                errors.append({
                    "point_id": None,
                    "source": src,
                    "error": "point_id gerekli",
                })
                continue
            effective = _watermark_past_days(
                con,
                point_id=point.point_id,
                source=src,
                role="previous_runs",
                lead_hours=requested_leads,
                requested_past_days=past_days,
            )
            groups.setdefault((src, effective), []).append(point)

    for (src, effective), grouped_points in groups.items():
        # The base implementation is a compatibility loop. Run those calls
        # here one by one so legacy providers retain per-point fault isolation.
        if type(provider).fetch_previous_runs_batch is WeatherProvider.fetch_previous_runs_batch:
            for point in grouped_points:
                try:
                    series = provider.fetch_previous_runs(
                        _to_geopoint(point),
                        variables,
                        past_days=effective,
                        previous_days=previous_days,
                        forecast_days=forecast_days,
                        models=[src] if src else None,
                    )
                    result = _series_result(con, series, past_days=effective)
                    total_rows += result["rows"]
                    ingested += 1
                except Exception as exc:  # noqa: BLE001 — isolate legacy provider cells
                    errors.append({
                        "point_id": point.point_id,
                        "source": src,
                        "error": str(exc),
                    })
            continue
        try:
            series_list = provider.fetch_previous_runs_batch(
                [_to_geopoint(point) for point in grouped_points],
                variables,
                past_days=effective,
                previous_days=previous_days,
                forecast_days=forecast_days,
                models=[src] if src else None,
            )
            if len(series_list) != len(grouped_points):
                raise ValueError(
                    "provider batch response count does not match point count "
                    f"({len(series_list)} != {len(grouped_points)})"
                )
            for series in series_list:
                result = _series_result(con, series, past_days=effective)
                total_rows += result["rows"]
                ingested += 1
        except Exception as exc:  # noqa: BLE001 — tolerate a failed HTTP batch
            errors.extend(
                {
                    "point_id": point.point_id,
                    "source": src,
                    "error": str(exc),
                }
                for point in grouped_points
            )
    return {"points": len(points), "ingested": ingested, "rows": total_rows, "errors": errors}
