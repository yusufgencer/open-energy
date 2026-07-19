from __future__ import annotations

from pathlib import Path

import duckdb

_SCHEMA = Path(__file__).with_name("schema.sql")
_WEATHER_RAW_NO_UNIQUE_MIGRATION = "20260719_weather_raw_no_unique"

# Idempotent column migrations for DBs created before a column existed.
# schema.sql's CREATE ... IF NOT EXISTS never alters an existing table, so new
# columns on persistent DBs must be added here. ADD COLUMN IF NOT EXISTS is a
# no-op on fresh DBs (the CREATE already added them).
_MIGRATIONS: list[tuple[str, str, str]] = [
    *[("experiment_trials", c, "DOUBLE") for c in (
        "test_coverage", "test_interval_width", "test_crps", "test_peak_bias", "test_peak_mae")],
    *[("strategy_trials", c, "DOUBLE") for c in (
        "test_coverage", "test_interval_width", "test_crps", "test_peak_bias", "test_peak_mae")],
    *[(table, c, "DOUBLE") for table in ("experiment_trials", "strategy_trials")
      for c in ("test_clean_nrmse", "clean_skill_score")],
    *[(table, c, "INTEGER") for table in ("experiment_trials", "strategy_trials")
      for c in ("train_all_count", "train_clean_count", "eval_all_count", "eval_clean_count")],
    ("strategy_trials", "band_status", "VARCHAR DEFAULT 'unknown'"),
    *[("strategy_trials", c, "DOUBLE") for c in (
        "sel_coverage", "sel_coverage_lower", "sel_coverage_upper", "sel_winkler",
        "sel_kupiec_pvalue", "sel_independence_pvalue",
        "sel_conditional_coverage_pvalue")],
    ("strategy_trials", "sel_n", "INTEGER"),
    ("strategy_trials", "champion_metric_name", "VARCHAR"),
    ("strategy_trials", "champion_metric_lower", "DOUBLE"),
    ("strategy_trials", "champion_metric_upper", "DOUBLE"),
    ("strategy_trials", "mcs_members", "VARCHAR"),
    ("strategy_trials", "uncertainty_diagnostics", "VARCHAR"),
    ("strategy_trials", "evaluation_grade", "VARCHAR DEFAULT 'exploratory'"),
    ("strategy_trials", "evaluation_n_days", "INTEGER"),
    ("strategy_trials", "evaluation_origins", "INTEGER"),
    ("strategy_trials", "evaluation_seasons", "INTEGER"),
    *[("scenario_runs", c, "DOUBLE") for c in ("crps", "peak_bias", "peak_mae")],
    ("scenario_runs", "source_policy", "VARCHAR"),
    ("scenario_runs", "target_policy", "VARCHAR"),
    ("scenario_runs", "evaluation_grade", "VARCHAR DEFAULT 'exploratory'"),
    ("scenario_runs", "evaluation_n_days", "INTEGER"),
    ("scenario_runs", "evaluation_origins", "INTEGER"),
    ("scenario_runs", "evaluation_seasons", "INTEGER"),
    ("plant_points", "grid_id", "VARCHAR"),
    ("plant_points", "grid_row", "INTEGER"),
    ("plant_points", "grid_col", "INTEGER"),
    ("drift_state", "last_window", "TIMESTAMP"),
    ("forecasts", "strategy_trial_id", "BIGINT"),
    ("jobs", "dedupe_key", "VARCHAR"),
    ("jobs", "payload", "VARCHAR"),
    # DuckDB does not support adding NOT NULL constraints via ALTER TABLE.
    # Fresh databases get the constraints from schema.sql; legacy databases get
    # the same operational defaults and runner-side validation.
    ("jobs", "attempt", "INTEGER DEFAULT 0"),
    ("jobs", "max_attempts", "INTEGER DEFAULT 3"),
    ("jobs", "available_at", "TIMESTAMP DEFAULT now()"),
    ("jobs", "lease_owner", "VARCHAR"),
    ("jobs", "lease_until", "TIMESTAMP"),
    ("jobs", "heartbeat_at", "TIMESTAMP"),
    ("jobs", "last_error", "VARCHAR"),
    ("jobs", "started_at", "TIMESTAMP"),
    ("jobs", "completed_at", "TIMESTAMP"),
    ("jobs", "updated_at", "TIMESTAMP DEFAULT now()"),
]

_SQL_MIGRATIONS = [
    """CREATE UNIQUE INDEX IF NOT EXISTS forecasts_issue_valid_horizon_uq
       ON forecasts (plant_id, point_id, horizon_hours, issue_time, valid_time)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS jobs_dedupe_key_uq ON jobs (dedupe_key)",
]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _migrate_weather_raw_no_unique(con: duckdb.DuckDBPyConnection) -> None:
    """Remove weather_raw's legacy ART-backed UNIQUE constraint atomically.

    DuckDB cannot drop this constraint in place.  Rebuilding from table_info
    preserves the exact column order, types, nullability, defaults, and data
    without copying the UNIQUE constraint.  The DDL and migration marker share
    one transaction, so an interrupted migration leaves the original table
    intact and is safe to retry.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id VARCHAR PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )
    applied = con.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_id=?",
        [_WEATHER_RAW_NO_UNIQUE_MIGRATION],
    ).fetchone()
    if applied is not None:
        return

    constraints = con.execute(
        """
        SELECT constraint_type
        FROM duckdb_constraints()
        WHERE table_name='weather_raw' AND constraint_type='UNIQUE'
        """
    ).fetchall()
    con.execute("BEGIN TRANSACTION")
    try:
        if constraints:
            columns = con.execute("PRAGMA table_info('weather_raw')").fetchall()
            definitions: list[str] = []
            names: list[str] = []
            for _, name, data_type, not_null, default, primary_key in columns:
                definition = f"{_quote_identifier(name)} {data_type}"
                if default is not None:
                    definition += f" DEFAULT {default}"
                if not_null:
                    definition += " NOT NULL"
                if primary_key:
                    definition += " PRIMARY KEY"
                definitions.append(definition)
                names.append(_quote_identifier(name))
            # A transaction rollback normally removes this table.  The explicit
            # cleanup also makes retries safe after a manually interrupted or
            # pre-versioned rebuild attempt.
            con.execute("DROP TABLE IF EXISTS weather_raw__t32_rebuild")
            con.execute(
                "CREATE TABLE weather_raw__t32_rebuild ("
                + ", ".join(definitions)
                + ")"
            )
            column_list = ", ".join(names)
            con.execute(
                f"INSERT INTO weather_raw__t32_rebuild ({column_list}) "
                f"SELECT {column_list} FROM weather_raw"
            )
            con.execute("DROP TABLE weather_raw")
            con.execute(
                "ALTER TABLE weather_raw__t32_rebuild RENAME TO weather_raw"
            )
        con.execute(
            "INSERT INTO schema_migrations (migration_id) VALUES (?)",
            [_WEATHER_RAW_NO_UNIQUE_MIGRATION],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    target = ":memory:" if db_path is None else str(db_path)
    return duckdb.connect(target)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA.read_text())
    _migrate_weather_raw_no_unique(con)
    for table, column, coltype in _MIGRATIONS:
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {coltype}")
    for statement in _SQL_MIGRATIONS:
        con.execute(statement)
