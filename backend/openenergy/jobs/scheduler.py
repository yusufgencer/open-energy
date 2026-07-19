"""Declarative APScheduler producer for OpenEnergy's durable job queue.

This module is intentionally a producer only. A scheduled tick inserts durable
jobs; it never imports or invokes a job handler. Execution belongs exclusively
to :mod:`openenergy.jobs.worker`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import duckdb
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from openenergy.config import Settings
from openenergy.jobs.runner import create_job, deterministic_dedupe_key


ConnectionFactory = Callable[[], duckdb.DuckDBPyConnection]
PayloadFactory = Callable[[duckdb.DuckDBPyConnection, datetime], Iterable[dict]]


@dataclass(frozen=True)
class ScheduleDefinition:
    name: str
    job_kind: str
    trigger: BaseTrigger
    period_seconds: int | None
    payloads: PayloadFactory


def _slot(definition: ScheduleDefinition, tick_time: datetime) -> str:
    """Return the schedule occurrence key used for durable deduplication."""
    aware = tick_time if tick_time.tzinfo is not None else tick_time.replace(tzinfo=timezone.utc)
    if definition.period_seconds is None:
        # The only non-interval default is monthly retraining.
        trigger_timezone = getattr(definition.trigger, "timezone", timezone.utc)
        return aware.astimezone(trigger_timezone).strftime("%Y-%m")
    seconds = int(aware.timestamp())
    return str(seconds - seconds % definition.period_seconds)


def _plants_with_point(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    return con.execute(
        """
        SELECT p.plant_id, p.kind, p.capacity_mw, pp.point_id
        FROM plants p
        JOIN plant_points pp ON pp.point_id = (
            SELECT point_id
            FROM plant_points
            WHERE plant_id = p.plant_id
            ORDER BY CASE WHEN grid_id IS NULL THEN 0 ELSE 1 END, point_id
            LIMIT 1
        )
        ORDER BY p.plant_id
        """
    ).fetchall()


def _base_payloads(settings: Settings) -> PayloadFactory:
    def payloads(con: duckdb.DuckDBPyConnection, _tick: datetime) -> Iterable[dict]:
        for plant_id, kind, capacity_mw, point_id in _plants_with_point(con):
            yield {
                "plant_id": plant_id,
                "point_id": int(point_id),
                "horizons": list(settings.scheduler_horizons),
                "capacity_mw": float(capacity_mw),
                "kind": kind,
            }

    return payloads


def build_schedule_registry(settings: Settings) -> tuple[ScheduleDefinition, ...]:
    """Build enabled schedules from settings, with timezone-aware triggers."""
    tz = ZoneInfo(settings.scheduler_timezone)
    base = _base_payloads(settings)
    schedules: list[ScheduleDefinition] = []

    if settings.scheduler_monthly_retrain_enabled:
        def retrain_payloads(con, tick):
            for payload in base(con, tick):
                yield {
                    **payload,
                    "nwp_sources": list(settings.scheduler_nwp_sources),
                    "n_trials": 10,
                    "n_splits": 4,
                    "embargo": 24,
                    "seed": 42,
                }

        schedules.append(ScheduleDefinition(
            name="monthly_retrain",
            job_kind="retrain",
            trigger=CronTrigger(
                day=settings.scheduler_retrain_day,
                hour=settings.scheduler_retrain_hour,
                minute=0,
                timezone=tz,
            ),
            period_seconds=None,
            payloads=retrain_payloads,
        ))

    if settings.scheduler_drift_enabled:
        def drift_payloads(con, tick):
            for payload in base(con, tick):
                yield {
                    **payload,
                    "nwp_sources": list(settings.scheduler_nwp_sources),
                    "n_trials": 10,
                    "n_splits": 4,
                    "embargo": 24,
                    "seed": 42,
                    "lookback_days": 30,
                    "min_samples": 12,
                    "breach_ratio": 1.5,
                    "recover_ratio": 1.2,
                    "min_consecutive": 2,
                    "psi_threshold": 0.25,
                    "ks_alpha": 0.05,
                    "freshness_hours": 36.0,
                }

        schedules.append(ScheduleDefinition(
            name="daily_drift",
            job_kind="drift_check",
            trigger=CronTrigger(hour=settings.scheduler_drift_hour, minute=0, timezone=tz),
            period_seconds=24 * 60 * 60,
            payloads=drift_payloads,
        ))

    if settings.scheduler_serving_enabled:
        def serving_payloads(con, tick):
            issue_time = tick.astimezone(timezone.utc).isoformat()
            for payload in base(con, tick):
                yield {**payload, "issue_time": issue_time, "models": None}

        period = settings.scheduler_serving_minutes * 60
        schedules.append(ScheduleDefinition(
            name="periodic_serving",
            job_kind="forecast",
            trigger=IntervalTrigger(minutes=settings.scheduler_serving_minutes, timezone=tz),
            period_seconds=period,
            payloads=serving_payloads,
        ))

    if settings.scheduler_reforecast_enabled:
        def reforecast_payloads(con, tick):
            issue_time = tick.astimezone(timezone.utc).isoformat()
            for payload in base(con, tick):
                for horizon in payload.pop("horizons"):
                    yield {
                        **payload,
                        "horizon_hours": horizon,
                        "issue_time": issue_time,
                        "w_new": 0.7,
                    }

        period = settings.scheduler_reforecast_minutes * 60
        schedules.append(ScheduleDefinition(
            name="periodic_reforecast",
            job_kind="reforecast",
            trigger=IntervalTrigger(minutes=settings.scheduler_reforecast_minutes, timezone=tz),
            period_seconds=period,
            payloads=reforecast_payloads,
        ))

    return tuple(schedules)


def scheduled_tick(
    connection_factory: ConnectionFactory,
    definition: ScheduleDefinition,
    *,
    tick_time: datetime | None = None,
) -> list[int]:
    """Enqueue all jobs for one occurrence and return their durable ids."""
    at = tick_time or datetime.now(timezone.utc)
    con = connection_factory()
    ids: list[int] = []
    try:
        for payload in definition.payloads(con, at):
            identity = {
                "schedule": definition.name,
                "slot": _slot(definition, at),
                "plant_id": payload["plant_id"],
                "horizon_hours": payload.get("horizon_hours"),
            }
            ids.append(create_job(
                con,
                definition.job_kind,
                f"schedule={definition.name} slot={identity['slot']}",
                payload=payload,
                dedupe_key=deterministic_dedupe_key(definition.job_kind, identity),
            ))
    finally:
        con.close()
    return ids


def build_scheduler(
    connection_factory: ConnectionFactory,
    settings: Settings,
) -> BackgroundScheduler:
    """Create (but do not start) the configured background scheduler."""
    scheduler = BackgroundScheduler(timezone=ZoneInfo(settings.scheduler_timezone))
    for definition in build_schedule_registry(settings):
        scheduler.add_job(
            scheduled_tick,
            trigger=definition.trigger,
            kwargs={"connection_factory": connection_factory, "definition": definition},
            id=definition.name,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    return scheduler
