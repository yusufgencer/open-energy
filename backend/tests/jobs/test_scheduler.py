from datetime import datetime, timezone
from unittest.mock import Mock

from openenergy.assets.repository import Plant, PlantPoint, add_point, create_plant
from openenergy.config import Settings
from openenergy.jobs.scheduler import build_schedule_registry, scheduled_tick


def _seed_plant(db):
    create_plant(
        db,
        Plant(
            plant_id="wf1",
            name="Wind Farm",
            kind="wind",
            capacity_mw=10,
            timezone="Europe/Istanbul",
        ),
    )
    add_point(
        db,
        PlantPoint(
            plant_id="wf1",
            point_type="weather",
            latitude=39.9,
            longitude=32.8,
        ),
    )


def _serving_schedule(settings):
    return next(
        definition
        for definition in build_schedule_registry(settings)
        if definition.name == "periodic_serving"
    )


def test_scheduled_tick_only_enqueues(db, monkeypatch):
    from openenergy.jobs.worker import DEFAULT_HANDLERS

    _seed_plant(db)
    handler = Mock()
    monkeypatch.setitem(DEFAULT_HANDLERS, "forecast", handler)
    definition = _serving_schedule(Settings(data_dir="/tmp/openenergy-scheduler-test"))

    ids = scheduled_tick(
        db.cursor,
        definition,
        tick_time=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert len(ids) == 1
    job = db.execute(
        "SELECT kind, status, payload FROM jobs WHERE job_id=?",
        [ids[0]],
    ).fetchone()
    assert job[0:2] == ("forecast", "pending")
    assert '"plant_id":"wf1"' in job[2]
    handler.assert_not_called()


def test_double_tick_dedupes(db):
    _seed_plant(db)
    definition = _serving_schedule(Settings(data_dir="/tmp/openenergy-scheduler-test"))
    at = datetime(2026, 7, 19, 12, 15, tzinfo=timezone.utc)

    first = scheduled_tick(db.cursor, definition, tick_time=at)
    second = scheduled_tick(db.cursor, definition, tick_time=at)

    assert second == first
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_registry_is_declarative_timezone_aware_and_individually_disableable():
    settings = Settings(
        data_dir="/tmp/openenergy-scheduler-test",
        scheduler_timezone="Europe/Istanbul",
        scheduler_drift_enabled=False,
        scheduler_reforecast_enabled=False,
    )

    registry = build_schedule_registry(settings)

    assert [definition.name for definition in registry] == [
        "monthly_retrain",
        "periodic_serving",
    ]
    assert all(str(definition.trigger.timezone) == "Europe/Istanbul" for definition in registry)
