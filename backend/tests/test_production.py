import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pytest

from openenergy.assets.repository import Plant, create_plant
from openenergy.ingestion.epias import ingest_epias_production
from openenergy.ingestion.production import load_production_csv, write_production

FIX = Path(__file__).parent / "fixtures"


def _prod(rows):
    return pl.DataFrame(
        {"ts": [r[0] for r in rows], "power_mw": [r[1] for r in rows]},
        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
    )


def test_reingest_is_idempotent(db):
    """T-07: re-ingesting the same (plant_id, ts) must not raise a PK violation and
    must leave exactly one row per timestamp (idempotent upsert)."""
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    frame = _prod([(dt.datetime(2024, 6, 1, 0), 3.0), (dt.datetime(2024, 6, 1, 1), 4.0)])
    write_production(db, "wf1", frame)
    write_production(db, "wf1", frame)  # same data again — must not blow up
    cnt = db.execute("SELECT count(*) FROM production WHERE plant_id='wf1'").fetchone()[0]
    assert cnt == 2


def test_revision_overwrites(db):
    """T-07: a revised value for the same (plant_id, ts) overwrites the prior one
    (EPİAŞ realtime → UEVM revision semantics)."""
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    ts = dt.datetime(2024, 6, 1, 0)
    write_production(db, "wf1", _prod([(ts, 3.0)]))
    write_production(db, "wf1", _prod([(ts, 5.5)]))  # revised
    rows = db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' AND ts=?", [ts]
    ).fetchall()
    assert len(rows) == 1 and rows[0][0] == 5.5


def test_frame_write_is_atomic_and_latest_revision_wins(db):
    """A frame is one atomic upsert, and duplicate timestamps use the last revision."""
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    ts = dt.datetime(2024, 6, 1, 0)
    write_production(db, "wf1", _prod([(ts, 3.0), (ts, 5.5)]))
    assert db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' AND ts=?", [ts]
    ).fetchone()[0] == 5.5

    invalid = pl.DataFrame(
        {"ts": [dt.datetime(2024, 6, 1, 1), None], "power_mw": [4.0, 6.0]},
        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
    )
    with pytest.raises(duckdb.ConstraintException, match="NOT NULL"):
        write_production(db, "wf1", invalid)
    assert db.execute(
        "SELECT count(*) FROM production WHERE plant_id='wf1'"
    ).fetchone()[0] == 1


def test_load_csv_utc_naive():
    frame = load_production_csv(FIX / "production_sample.csv")
    assert frame.columns == ["ts", "power_mw"]
    assert frame.height == 3
    assert frame["ts"][0] == dt.datetime(2024, 6, 1, 0, 0)
    assert frame["power_mw"][1] == 4.1


def test_load_csv_converts_timezone_to_utc():
    # Europe/Istanbul (+03:00) → UTC: 03:00 yerel == 00:00 UTC
    frame = load_production_csv(FIX / "production_sample.csv", timezone="Europe/Istanbul")
    assert frame["ts"][0] == dt.datetime(2024, 5, 31, 21, 0)


def test_write_production_persists(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    frame = load_production_csv(FIX / "production_sample.csv")
    n = write_production(db, "wf1", frame)
    assert n == 3
    cnt = db.execute("SELECT count(*) FROM production WHERE plant_id='wf1'").fetchone()[0]
    assert cnt == 3


def test_negative_is_flagged_not_clipped(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    ts = dt.datetime(2024, 6, 1, 0)

    write_production(db, "wf1", _prod([(ts, -0.25)]), source="test")

    assert db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' AND ts=?", [ts]
    ).fetchone()[0] == -0.25
    flag = db.execute(
        "SELECT flag_code, severity, observed_power_mw, source "
        "FROM production_quality WHERE plant_id='wf1' AND ts=?",
        [ts],
    ).fetchone()
    assert flag == ("NEGATIVE_POWER", "error", -0.25, "test")


def test_over_capacity_respects_explicit_tolerance(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    t0 = dt.datetime(2024, 6, 1, 0)

    write_production(
        db,
        "wf1",
        _prod([(t0, 10.1), (t0 + dt.timedelta(hours=1), 10.21)]),
        capacity_tolerance=0.02,
    )

    rows = db.execute(
        "SELECT ts, flag_code FROM production_quality "
        "WHERE plant_id='wf1' AND flag_code='OVER_CAPACITY' ORDER BY ts"
    ).fetchall()
    assert rows == [(t0 + dt.timedelta(hours=1), "OVER_CAPACITY")]
    # Flagging must not alter either raw value.
    assert db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' ORDER BY ts"
    ).fetchall() == [(10.1,), (10.21,)]


def test_stale_run_flagged(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    t0 = dt.datetime(2024, 6, 1, 0)
    frame = _prod([(t0 + dt.timedelta(hours=i), 4.25) for i in range(6)])

    write_production(db, "wf1", frame)

    flags = db.execute(
        "SELECT ts FROM production_quality "
        "WHERE plant_id='wf1' AND flag_code='STALE_RUN' ORDER BY ts"
    ).fetchall()
    assert flags == [(t0 + dt.timedelta(hours=i),) for i in range(6)]
    assert db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' ORDER BY ts"
    ).fetchall() == [(4.25,)] * 6


def test_stale_zero_run_is_not_inferred_as_sensor_freeze(db):
    """Zero runs can be real outages or solar night, so production-only QC abstains."""
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    t0 = dt.datetime(2024, 6, 1, 0)
    write_production(
        db, "wf1", _prod([(t0 + dt.timedelta(hours=i), 0.0) for i in range(8)])
    )
    assert db.execute(
        "SELECT count(*) FROM production_quality "
        "WHERE plant_id='wf1' AND flag_code='STALE_RUN'"
    ).fetchone()[0] == 0


def test_revision_replaces_obsolete_quality_flags(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    ts = dt.datetime(2024, 6, 1, 0)
    write_production(db, "wf1", _prod([(ts, -1.0)]), revision="r1")
    write_production(db, "wf1", _prod([(ts, 5.0)]), revision="r2")

    assert db.execute(
        "SELECT power_mw FROM production WHERE plant_id='wf1' AND ts=?", [ts]
    ).fetchone()[0] == 5.0
    assert db.execute(
        "SELECT count(*) FROM production_quality WHERE plant_id='wf1' AND ts=?", [ts]
    ).fetchone()[0] == 0


def test_epias_ingest_runs_qc_automatically(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))

    class Client:
        def post(self, path, body):
            assert path == "/generation/data/realtime-generation"
            return {
                "items": [{
                    "date": "2024-06-01T03:00:00+03:00",
                    "total": -0.4,
                }]
            }

    ingest_epias_production(
        db,
        "wf1",
        start=dt.date(2024, 6, 1),
        end=dt.date(2024, 6, 1),
        power_plant_id=42,
        client=Client(),
    )

    assert db.execute(
        "SELECT flag_code, source, revision FROM production_quality "
        "WHERE plant_id='wf1'"
    ).fetchone() == (
        "NEGATIVE_POWER",
        "epias:realtime-generation:total",
        "2024-06-01_2024-06-01",
    )
