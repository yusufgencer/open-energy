from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

DEFAULT_CAPACITY_TOLERANCE = 0.02
STALE_MIN_OBSERVATIONS = 6
STALE_MIN_DURATION = dt.timedelta(hours=5)


def load_production_csv(path: str | Path, *, timestamp_col: str = "timestamp",
                        power_col: str = "power_mw", timezone: str = "UTC") -> pl.DataFrame:
    raw = pl.read_csv(path)
    if timestamp_col not in raw.columns or power_col not in raw.columns:
        raise ValueError(f"CSV '{timestamp_col}' ve '{power_col}' kolonlarını içermeli")
    ts = pl.col(timestamp_col).str.to_datetime(time_unit="us")
    if timezone != "UTC":
        ts = ts.dt.replace_time_zone(timezone).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    frame = raw.select(
        ts.alias("ts"),
        pl.col(power_col).cast(pl.Float64).alias("power_mw"),
    )
    if frame["power_mw"].is_null().any():
        raise ValueError("power_mw sayısal olmayan/eksik değer içeriyor")
    return frame


def _revision(source: str, ts: dt.datetime, power_mw: float, revision: str | None) -> str:
    if revision is not None:
        return str(revision)
    raw = f"{source}|{ts.isoformat()}|{power_mw:.17g}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _stale_timestamps(
    rows: Iterable[tuple[dt.datetime, float]],
    *,
    capacity_mw: float,
    capacity_tolerance: float,
) -> set[dt.datetime]:
    """Return conservative frozen-meter runs.

    Only exactly-equal, positive, sub-capacity values sampled at a regular
    cadence and spanning at least five hours qualify. Zero is intentionally
    excluded (real outage / solar night) and nameplate plateaus are excluded
    (legitimate clipping).
    """
    ordered = sorted(rows)
    flagged: set[dt.datetime] = set()
    run: list[tuple[dt.datetime, float]] = []
    cadence: dt.timedelta | None = None

    def finish() -> None:
        if (
            len(run) >= STALE_MIN_OBSERVATIONS
            and run[-1][0] - run[0][0] >= STALE_MIN_DURATION
        ):
            flagged.update(ts for ts, _ in run)

    for ts, value in ordered:
        eligible = 0.0 < value < capacity_mw * (1.0 + capacity_tolerance)
        if not eligible:
            finish()
            run, cadence = [], None
            continue
        if not run:
            run = [(ts, value)]
            cadence = None
            continue
        gap = ts - run[-1][0]
        same_value = value == run[-1][1]
        regular = dt.timedelta(0) < gap <= dt.timedelta(hours=1) and (
            cadence is None or gap == cadence
        )
        if same_value and regular:
            if cadence is None:
                cadence = gap
            run.append((ts, value))
        else:
            finish()
            run, cadence = [(ts, value)], None
    finish()
    return flagged


def _quality_rows(
    plant_id: str,
    rows: list[tuple[str, dt.datetime, float]],
    *,
    capacity_mw: float,
    capacity_tolerance: float,
    source: str,
    revision: str | None,
) -> list[tuple]:
    quality: list[tuple] = []
    for _, ts, value in rows:
        rev = _revision(source, ts, value, revision)
        if value < 0:
            quality.append((
                plant_id, ts, "NEGATIVE_POWER", "error",
                json.dumps({"observed_power_mw": value}, sort_keys=True),
                source, rev, value,
            ))
        if value > capacity_mw * (1.0 + capacity_tolerance):
            quality.append((
                plant_id, ts, "OVER_CAPACITY", "error",
                json.dumps({
                    "capacity_mw": capacity_mw,
                    "observed_power_mw": value,
                    "tolerance_fraction": capacity_tolerance,
                    "threshold_mw": capacity_mw * (1.0 + capacity_tolerance),
                }, sort_keys=True),
                source, rev, value,
            ))
    return quality


def write_production(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    frame: pl.DataFrame,
    *,
    source: str = "unknown",
    revision: str | None = None,
    capacity_tolerance: float = DEFAULT_CAPACITY_TOLERANCE,
) -> int:
    """Atomically upsert raw production and its current QC flags.

    QC never clips, replaces, or deletes the raw value. ``capacity_tolerance``
    is an explicit fraction (0.02 means 2% above plant nameplate).
    """
    if capacity_tolerance < 0:
        raise ValueError("capacity_tolerance negatif olamaz")
    enriched = frame.select(
        pl.lit(plant_id).alias("plant_id"), "ts", "power_mw"
    )
    if enriched.height == 0:
        return 0

    # Keep the latest observation when a provider returns more than one revision
    # for the same timestamp in a single response.
    rows = enriched.unique(
        subset=["plant_id", "ts"], keep="last", maintain_order=True
    ).rows()
    plant = con.execute(
        "SELECT capacity_mw FROM plants WHERE plant_id=?", [plant_id]
    ).fetchone()
    if plant is None:
        raise ValueError(f"Bilinmeyen plant_id: {plant_id}")
    capacity_mw = float(plant[0])
    if capacity_mw <= 0:
        raise ValueError("plant capacity_mw > 0 olmalı")

    placeholders = ", ".join(["(?, ?, ?)"] * len(rows))
    affected_ts = [row[1] for row in rows]

    con.execute("BEGIN TRANSACTION")
    try:
        # One statement preserves T-07's all-or-nothing frame upsert.
        con.execute(
            "INSERT OR REPLACE INTO production (plant_id, ts, power_mw) VALUES "
            + placeholders,
            [value for row in rows for value in row],
        )
        # Build annotations only after the database has validated the raw frame,
        # preserving the original constraint error and rollback semantics.
        qc_rows = _quality_rows(
            plant_id, rows, capacity_mw=capacity_mw,
            capacity_tolerance=capacity_tolerance, source=source, revision=revision,
        )

        # A revision can make an old point-level flag disappear.
        ts_placeholders = ", ".join(["?"] * len(affected_ts))
        con.execute(
            "DELETE FROM production_quality "
            f"WHERE plant_id=? AND ts IN ({ts_placeholders}) "
            "AND flag_code IN ('NEGATIVE_POWER','OVER_CAPACITY')",
            [plant_id, *affected_ts],
        )
        if qc_rows:
            qc_placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(qc_rows))
            con.execute(
                "INSERT OR REPLACE INTO production_quality "
                "(plant_id, ts, flag_code, severity, details, source, revision, "
                "observed_power_mw) VALUES " + qc_placeholders,
                [value for row in qc_rows for value in row],
            )

        # Recompute frozen runs for this plant after the upsert. This handles a
        # run split across ingest batches and revisions that break an old run.
        all_rows = con.execute(
            "SELECT ts, power_mw FROM production WHERE plant_id=? ORDER BY ts",
            [plant_id],
        ).fetchall()
        stale_ts = _stale_timestamps(
            all_rows, capacity_mw=capacity_mw,
            capacity_tolerance=capacity_tolerance,
        )
        con.execute(
            "DELETE FROM production_quality "
            "WHERE plant_id=? AND flag_code='STALE_RUN'",
            [plant_id],
        )
        if stale_ts:
            values = {ts: float(value) for ts, value in all_rows}
            stale_rows = [
                (
                    plant_id, ts, "STALE_RUN", "warning",
                    json.dumps({
                        "minimum_duration_hours": STALE_MIN_DURATION.total_seconds() / 3600,
                        "minimum_observations": STALE_MIN_OBSERVATIONS,
                        "observed_power_mw": values[ts],
                    }, sort_keys=True),
                    "qc:stale-run",
                    _revision("qc:stale-run", ts, values[ts], None),
                    values[ts],
                )
                for ts in sorted(stale_ts)
            ]
            stale_placeholders = ", ".join(
                ["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(stale_rows)
            )
            con.execute(
                "INSERT INTO production_quality "
                "(plant_id, ts, flag_code, severity, details, source, revision, "
                "observed_power_mw) VALUES " + stale_placeholders,
                [value for row in stale_rows for value in row],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return frame.height
