from __future__ import annotations

import datetime as dt

import duckdb
import polars as pl

from openenergy.config import Settings, get_settings
from openenergy.providers.base import WeatherSeries

# Canonical schema for the columns returned by read_weather
_READ_WEATHER_SCHEMA: dict[str, pl.PolarsDataType] = {
    "valid_time": pl.Datetime("us"),
    "issue_time": pl.Datetime("us"),
    "lead_hours": pl.Int32,
    "variable": pl.Utf8,
    "value": pl.Float64,
    "model": pl.Utf8,
}


# Rows per bulk INSERT statement. A single multi-row VALUES executes as one
# query (one plan, batched constraint check) — far faster than executemany's
# per-row execution — while staying pyarrow-free (register() needs pyarrow).
_BULK_CHUNK = 500


def _retention_days(role: str, settings: Settings) -> int:
    field = {
        "forecast": "weather_retention_forecast_days",
        "archive": "weather_retention_archive_days",
        "previous_runs": "weather_retention_previous_runs_days",
        "ensemble": "weather_retention_ensemble_days",
        "single_runs": "weather_retention_single_runs_days",
    }.get(role)
    if field is None:
        raise ValueError(f"bilinmeyen weather role: {role}")
    return int(getattr(settings, field))


def prune_weather_raw(
    con: duckdb.DuckDBPyConnection,
    *,
    role: str,
    source: str,
    retention_days: int | None = None,
    as_of: dt.datetime | None = None,
    settings: Settings | None = None,
) -> int:
    """Prune one role/source partition by valid_time and return deleted rows.

    With no explicit ``as_of``, the partition high-water mark anchors the
    window.  This makes historical backfills deterministic and avoids deleting
    a complete historical batch merely because it was ingested later.
    """
    settings = settings or get_settings()
    if retention_days is None:
        if not settings.weather_retention_enabled:
            return 0
        retention_days = _retention_days(role, settings)
    if retention_days <= 0:
        raise ValueError("retention_days pozitif olmalı")
    anchor = as_of
    if anchor is None:
        row = con.execute(
            "SELECT max(valid_time) FROM weather_raw WHERE role=? AND model=?",
            [role, source],
        ).fetchone()
        anchor = row[0] if row else None
    if anchor is None:
        return 0
    if anchor.tzinfo is not None:
        anchor = anchor.replace(tzinfo=None)
    cutoff = anchor - dt.timedelta(days=retention_days)
    before = con.execute(
        "SELECT count(*) FROM weather_raw "
        "WHERE role=? AND model=? AND valid_time < ?",
        [role, source, cutoff],
    ).fetchone()[0]
    if before:
        con.execute(
            "DELETE FROM weather_raw "
            "WHERE role=? AND model=? AND valid_time < ?",
            [role, source, cutoff],
        )
    return int(before)


def write_weather_series(
    con: duckdb.DuckDBPyConnection,
    series: WeatherSeries,
    *,
    retention_days: int | None = None,
    settings: Settings | None = None,
) -> int:
    frame = series.frame
    if frame.height == 0:
        return 0
    enriched = frame.select(
        pl.lit(series.point_id).cast(pl.Int64).alias("point_id"),
        pl.lit(series.role).alias("role"),
        pl.lit(series.model).alias("model"),
        "valid_time", "issue_time", "lead_hours", "variable", "value",
    )
    # Keep the final revision if a provider returns the same natural key more
    # than once. issue_time/value are revision payload, not key columns.
    rows_by_key: dict[tuple, tuple] = {}
    for row in enriched.rows():
        key = (row[0], row[1], row[2], row[3], row[5], row[6])
        rows_by_key[key] = row
    rows = list(rows_by_key.values())

    con.execute("BEGIN TRANSACTION")
    try:
        # Delete exact incoming natural keys in bulk. IS NOT DISTINCT FROM is
        # required because lead_hours is nullable for forecast/archive roles.
        for start in range(0, len(rows), _BULK_CHUNK):
            chunk = rows[start:start + _BULK_CHUNK]
            keys = [(r[0], r[1], r[2], r[3], r[5], r[6]) for r in chunk]
            placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(keys))
            params = [value for key in keys for value in key]
            con.execute(
                "DELETE FROM weather_raw AS existing USING "
                f"(VALUES {placeholders}) "
                "AS incoming(point_id, role, model, valid_time, lead_hours, variable) "
                "WHERE existing.point_id=incoming.point_id "
                "AND existing.role=incoming.role "
                "AND existing.model=incoming.model "
                "AND existing.valid_time=incoming.valid_time "
                "AND existing.lead_hours IS NOT DISTINCT FROM incoming.lead_hours "
                "AND existing.variable=incoming.variable",
                params,
            )

        for start in range(0, len(rows), _BULK_CHUNK):
            chunk = rows[start:start + _BULK_CHUNK]
            placeholders = ",".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            params = [value for row in chunk for value in row]
            con.execute(
                "INSERT INTO weather_raw "
                "(point_id, role, model, valid_time, issue_time, lead_hours, variable, value) "
                f"VALUES {placeholders}",
                params,
            )
        prune_weather_raw(
            con,
            role=series.role,
            source=series.model,
            retention_days=retention_days,
            settings=settings,
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return frame.height


def read_weather(con: duckdb.DuckDBPyConnection, point_id: int, role: str) -> pl.DataFrame:
    result = con.execute(
        """
        SELECT valid_time, issue_time, lead_hours, variable, value, model
        FROM weather_raw WHERE point_id=? AND role=?
        ORDER BY variable, valid_time
        """,
        [point_id, role],
    )
    cols = [d[0] for d in result.description]
    rows = result.fetchall()
    if not rows:
        return pl.DataFrame(schema={c: _READ_WEATHER_SCHEMA.get(c, pl.Utf8) for c in cols})
    df = pl.DataFrame([dict(zip(cols, row)) for row in rows])
    return df.with_columns([
        pl.col(c).cast(dtype) for c, dtype in _READ_WEATHER_SCHEMA.items() if c in df.columns
    ])
