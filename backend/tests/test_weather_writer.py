import datetime as dt

import polars as pl

from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import (
    prune_weather_raw,
    read_weather,
    write_weather_series,
)
from openenergy.storage import connect, init_schema


def _series():
    frame = pl.DataFrame(
        {
            "valid_time": [dt.datetime(2024, 6, 1), dt.datetime(2024, 6, 1, 1)],
            "issue_time": [None, None],
            "lead_hours": [None, None],
            "variable": ["wind_speed_100m", "wind_speed_100m"],
            "value": [7.2, 8.1],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    return WeatherSeries(point_id=1, role=ApiRole.FORECAST.value, model="icon", frame=frame)


def _prev_series():
    """Previous-runs series with non-null lead_hours for dedup tests."""
    rows = []
    for vt in [dt.datetime(2024, 6, 2, 0), dt.datetime(2024, 6, 2, 1)]:
        rows.append((vt, vt - dt.timedelta(hours=24), 24, "wind_speed_100m", 8.5))
    frame = pl.DataFrame(
        {
            "valid_time": [r[0] for r in rows],
            "issue_time": [r[1] for r in rows],
            "lead_hours": [r[2] for r in rows],
            "variable": [r[3] for r in rows],
            "value": [r[4] for r in rows],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    return WeatherSeries(point_id=5, role=ApiRole.PREVIOUS_RUNS.value,
                         model="best_match", frame=frame)


def test_write_returns_rowcount_and_persists(db):
    n = write_weather_series(db, _series())
    assert n == 2
    total = db.execute("SELECT count(*) FROM weather_raw").fetchone()[0]
    assert total == 2


def test_write_empty_series_is_noop(db):
    empty = WeatherSeries.empty(1, ApiRole.FORECAST.value, "icon")
    assert write_weather_series(db, empty) == 0


def test_read_weather_roundtrip(db):
    write_weather_series(db, _series())
    out = read_weather(db, point_id=1, role=ApiRole.FORECAST.value)
    assert out.height == 2
    assert set(out.columns) >= {"valid_time", "variable", "value"}


# --- Finding 3 & 4: canonical dtypes preserved for both populated and empty results ---

def test_read_weather_canonical_dtypes_populated(db):
    write_weather_series(db, _series())
    out = read_weather(db, point_id=1, role=ApiRole.FORECAST.value)
    assert out.height == 2
    assert out["valid_time"].dtype == pl.Datetime("us")
    assert out["lead_hours"].dtype == pl.Int32


def test_read_weather_canonical_dtypes_empty(db):
    out = read_weather(db, point_id=99, role=ApiRole.FORECAST.value)
    assert out.height == 0
    assert out["valid_time"].dtype == pl.Datetime("us")
    assert out["lead_hours"].dtype == pl.Int32


# --- Finding 5: no duplicate rows on re-ingestion of previous_runs series ---

def test_write_weather_series_no_duplicates_on_reingest(db):
    s = _prev_series()
    write_weather_series(db, s)
    write_weather_series(db, s)  # second identical write
    count = db.execute("SELECT count(*) FROM weather_raw WHERE point_id=5").fetchone()[0]
    assert count == s.frame.height  # must not double


def test_reingest_is_idempotent_and_keeps_latest_revision(db):
    original = _series()
    write_weather_series(db, original)
    revised = WeatherSeries(
        point_id=original.point_id,
        role=original.role,
        model=original.model,
        frame=original.frame.with_columns((pl.col("value") + 10).alias("value")),
    )

    write_weather_series(db, revised)
    write_weather_series(db, revised)

    rows = db.execute(
        """
        SELECT valid_time, value
        FROM weather_raw
        WHERE point_id=1 AND role='forecast' AND model='icon'
        ORDER BY valid_time
        """
    ).fetchall()
    assert rows == [
        (dt.datetime(2024, 6, 1), 17.2),
        (dt.datetime(2024, 6, 1, 1), 18.1),
    ]


def test_bulk_write_spans_multiple_chunks_and_dedups(db):
    # More rows than one bulk chunk (500): distinct valid_times, exercises chunking.
    n = 1300
    base = dt.datetime(2024, 6, 3)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    frame = pl.DataFrame(
        {
            "valid_time": vts,
            "issue_time": [vt - dt.timedelta(hours=24) for vt in vts],
            "lead_hours": [24] * n,
            "variable": ["wind_speed_100m"] * n,
            "value": [float(i) for i in range(n)],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    s = WeatherSeries(point_id=11, role=ApiRole.PREVIOUS_RUNS.value, model="icon", frame=frame)
    assert write_weather_series(db, s) == n
    assert db.execute("SELECT count(*) FROM weather_raw WHERE point_id=11").fetchone()[0] == n
    write_weather_series(db, s)  # re-ingest: still no duplicates across chunks
    assert db.execute("SELECT count(*) FROM weather_raw WHERE point_id=11").fetchone()[0] == n


def test_retention_prunes_rows_older_than_window(db):
    base = dt.datetime(2024, 6, 1)
    db.execute(
        "INSERT INTO weather_raw VALUES "
        "(99, 'archive', 'other_source', ?, NULL, NULL, 'temperature_2m', 5.0)",
        [base - dt.timedelta(days=500)],
    )
    frame = pl.DataFrame(
        {
            "valid_time": [base, base + dt.timedelta(days=29), base + dt.timedelta(days=31)],
            "issue_time": [None, None, None],
            "lead_hours": [None, None, None],
            "variable": ["temperature_2m"] * 3,
            "value": [10.0, 20.0, 30.0],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    series = WeatherSeries(
        point_id=22, role=ApiRole.ARCHIVE.value, model="era5", frame=frame)

    write_weather_series(db, series, retention_days=30)

    remaining = db.execute(
        "SELECT valid_time FROM weather_raw "
        "WHERE role='archive' AND model='era5' ORDER BY valid_time"
    ).fetchall()
    assert remaining == [
        (base + dt.timedelta(days=29),),
        (base + dt.timedelta(days=31),),
    ]
    assert db.execute(
        "SELECT count(*) FROM weather_raw "
        "WHERE role='archive' AND model='other_source'"
    ).fetchone()[0] == 1
    assert prune_weather_raw(
        db,
        role="archive",
        source="era5",
        retention_days=30,
    ) == 0


def test_failed_revision_rolls_back_delete_and_insert(db):
    write_weather_series(db, _series())
    invalid = _series().frame.with_columns(
        pl.when(pl.col("valid_time") == dt.datetime(2024, 6, 1, 1))
        .then(None)
        .otherwise(pl.col("valid_time"))
        .alias("valid_time"),
        (pl.col("value") + 100).alias("value"),
    )
    revised = WeatherSeries(
        point_id=1,
        role=ApiRole.FORECAST.value,
        model="icon",
        frame=invalid,
    )

    try:
        write_weather_series(db, revised)
    except Exception:
        pass
    else:
        raise AssertionError("NOT NULL valid_time insert should fail")

    values = db.execute(
        "SELECT value FROM weather_raw WHERE point_id=1 ORDER BY valid_time"
    ).fetchall()
    assert values == [(7.2,), (8.1,)]


def test_init_schema_rebuilds_legacy_weather_table_without_unique(tmp_path):
    db_path = tmp_path / "legacy.duckdb"
    con = connect(db_path)
    con.execute(
        """
        CREATE TABLE weather_raw (
            point_id BIGINT NOT NULL,
            role VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            valid_time TIMESTAMP NOT NULL,
            issue_time TIMESTAMP,
            lead_hours INTEGER,
            variable VARCHAR NOT NULL,
            value DOUBLE,
            UNIQUE (point_id, role, model, valid_time, lead_hours, variable)
        )
        """
    )
    con.execute(
        "INSERT INTO weather_raw VALUES "
        "(1, 'forecast', 'icon', '2024-01-01', NULL, NULL, 'temperature_2m', 3.5)"
    )
    before_columns = con.execute("PRAGMA table_info('weather_raw')").fetchall()

    init_schema(con)
    init_schema(con)

    after_columns = con.execute("PRAGMA table_info('weather_raw')").fetchall()
    unique_constraints = con.execute(
        "SELECT count(*) FROM duckdb_constraints() "
        "WHERE table_name='weather_raw' AND constraint_type='UNIQUE'"
    ).fetchone()[0]
    assert after_columns == before_columns
    assert unique_constraints == 0
    assert con.execute("SELECT count(*) FROM weather_raw").fetchone()[0] == 1
    assert con.execute(
        "SELECT count(*) FROM schema_migrations "
        "WHERE migration_id='20260719_weather_raw_no_unique'"
    ).fetchone()[0] == 1
    con.close()
