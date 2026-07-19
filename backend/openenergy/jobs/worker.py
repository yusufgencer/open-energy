from __future__ import annotations

import argparse
import multiprocessing
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import duckdb

from openenergy.assets.repository import list_points
from openenergy.forecasting.backtest import backfill_forecasts
from openenergy.forecasting.predict import run_serving_cycle
from openenergy.forecasting.reforecast import reforecast
from openenergy.ingestion.epias import ingest_epias_production
from openenergy.ingestion.grid import list_grid_points
from openenergy.ingestion.weather import ingest_grid_previous_runs, ingest_previous_runs
from openenergy.integrations import epias_store
from openenergy.jobs.runner import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_LEASE_SECONDS,
    claim_job,
    complete_claimed_job,
    enqueue_emergency_retrain,
    fail_claimed_job,
    heartbeat_job,
    recover_stale_jobs,
)
from openenergy.retraining.champion import retrain
from openenergy.retraining.drift import check_drift
from openenergy.retraining.postprocessor import fit_postprocessor
from openenergy.storage import connect, init_schema

JobHandler = Callable[[duckdb.DuckDBPyConnection, Mapping[str, Any], Path], Any]


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _handle_ingest_weather(con, p, _models_dir):
    point = next(
        point for point in list_points(con, p["plant_id"]) if point.point_id == p["point_id"]
    )
    return ingest_previous_runs(
        con,
        point,
        p["kind"],
        past_days=p["past_days"],
        previous_days=p["previous_days"],
    )["rows"]


def _handle_ingest_weather_grid(con, p, _models_dir):
    points = list_grid_points(con, p["plant_id"], p["grid_id"])
    return ingest_grid_previous_runs(
        con,
        points,
        p["kind"],
        sources=p.get("sources"),
        past_days=p["past_days"],
        previous_days=p["previous_days"],
    )["rows"]


def _handle_ingest_epias(con, p, _models_dir):
    client = epias_store.get_client()
    return ingest_epias_production(
        con,
        p["plant_id"],
        start=_date(p["start"]),
        end=_date(p["end"]),
        power_plant_id=p.get("power_plant_id"),
        plant_name=p.get("plant_name"),
        field=p["field"],
        client=client,
    )["rows"]


def _handle_experiment(con, p, models_dir):
    from openenergy.experiments.runner import run_experiment

    return run_experiment(
        con,
        plant_id=p["plant_id"],
        point_id=p["point_id"],
        horizon_hours_list=p["horizons"],
        capacity_mw=p["capacity_mw"],
        kind=p["kind"],
        nwp_sources=p["nwp_sources"],
        n_trials=p["n_trials"],
        top_k=p["top_k"],
        models_dir=models_dir / f"plant_{p['plant_id']}",
        point_ids=p.get("point_ids"),
        source=p.get("source"),
    )


def _handle_retrain(con, p, models_dir):
    return retrain(
        con,
        plant_id=p["plant_id"],
        point_id=p["point_id"],
        horizon_hours_list=p["horizons"],
        capacity_mw=p["capacity_mw"],
        kind=p["kind"],
        nwp_sources=p["nwp_sources"],
        n_trials=p["n_trials"],
        models_dir=models_dir / f"plant_{p['plant_id']}",
        n_splits=p["n_splits"],
        embargo=p["embargo"],
        seed=p["seed"],
    )


def _handle_reforecast(con, p, _models_dir):
    return reforecast(
        con,
        plant_id=p["plant_id"],
        point_id=p["point_id"],
        horizon_hours=p["horizon_hours"],
        capacity_mw=p["capacity_mw"],
        kind=p["kind"],
        issue_time=_dt(p["issue_time"]),
        w_new=p["w_new"],
    )


def _handle_forecast(con, p, _models_dir):
    point = next(
        point for point in list_points(con, p["plant_id"]) if point.point_id == p["point_id"]
    )
    return run_serving_cycle(
        con,
        plant_id=p["plant_id"],
        point=point,
        horizons=p["horizons"],
        capacity_mw=p["capacity_mw"],
        kind=p["kind"],
        issue_time=_dt(p["issue_time"]),
        models=p.get("models"),
    )


def _handle_forecast_backfill(con, p, _models_dir):
    return backfill_forecasts(
        con,
        plant_id=p["plant_id"],
        point_id=p["point_id"],
        horizons=p["horizons"],
        capacity_mw=p["capacity_mw"],
        kind=p["kind"],
        start_time=_dt(p.get("start_time")),
        end_time=_dt(p.get("end_time")),
        lookback_days=p["lookback_days"],
    )


def _handle_postprocessor_fit(con, p, _models_dir):
    return {
        horizon: fit_postprocessor(
            con,
            plant_id=p["plant_id"],
            horizon_hours=horizon,
            degree=p["degree"],
            alpha=p["alpha"],
            min_samples=p["min_samples"],
            lookback_days=p["lookback_days"],
        )
        for horizon in p["horizons"]
    }


def _handle_drift_check(con, p, _models_dir):
    """Evaluate drift in the worker; any resulting retrain is queued, not run."""

    def enqueue(plant_id: str, horizon_hours: int) -> int:
        return enqueue_emergency_retrain(
            con,
            plant_id,
            horizon_hours,
            payload={
                "plant_id": plant_id,
                "point_id": p["point_id"],
                "horizons": [horizon_hours],
                "capacity_mw": p["capacity_mw"],
                "kind": p["kind"],
                "nwp_sources": p["nwp_sources"],
                "n_trials": p["n_trials"],
                "n_splits": p["n_splits"],
                "embargo": p["embargo"],
                "seed": p["seed"],
            },
        )

    return [
        check_drift(
            con,
            plant_id=p["plant_id"],
            horizon_hours=horizon,
            capacity_mw=p["capacity_mw"],
            lookback_days=p["lookback_days"],
            min_samples=p["min_samples"],
            breach_ratio=p["breach_ratio"],
            recover_ratio=p["recover_ratio"],
            min_consecutive=p["min_consecutive"],
            enqueue=enqueue,
            point_id=p["point_id"],
            nwp_sources=p.get("nwp_sources"),
            psi_threshold=p.get("psi_threshold", 0.25),
            ks_alpha=p.get("ks_alpha", 0.05),
            freshness_hours=p.get("freshness_hours", 36.0),
        )
        for horizon in p["horizons"]
    ]


DEFAULT_HANDLERS: dict[str, JobHandler] = {
    "ingest_weather": _handle_ingest_weather,
    "ingest_weather_grid": _handle_ingest_weather_grid,
    "ingest_epias": _handle_ingest_epias,
    "experiment": _handle_experiment,
    "retrain": _handle_retrain,
    "emergency_retrain": _handle_retrain,
    "reforecast": _handle_reforecast,
    "forecast": _handle_forecast,
    "forecast_backfill": _handle_forecast_backfill,
    "postprocessor_fit": _handle_postprocessor_fit,
    "drift_check": _handle_drift_check,
}


def _execute_isolated(
    db_path: str,
    models_dir: str,
    handler: JobHandler,
    claimed: Mapping[str, Any],
    lease_seconds: int,
    heartbeat_seconds: float,
) -> Any:
    """ProcessPool target: every job opens and closes its own DB connection."""
    con = connect(db_path)
    stopped = threading.Event()

    def keep_lease() -> None:
        heartbeat_con = con.cursor()
        while not stopped.wait(heartbeat_seconds):
            heartbeat_job(
                heartbeat_con,
                int(claimed["job_id"]),
                str(claimed["lease_owner"]),
                lease_seconds=lease_seconds,
            )

    heartbeat = threading.Thread(target=keep_lease, daemon=True)
    heartbeat.start()
    try:
        return handler(con, claimed["payload"], Path(models_dir))
    finally:
        stopped.set()
        heartbeat.join()
        con.close()


def run_worker(
    db_path: Path | str,
    *,
    handlers: Mapping[str, JobHandler] | None = None,
    models_dir: Path | str | None = None,
    max_workers: int = 2,
    max_in_flight: int | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    heartbeat_seconds: float = 10.0,
    poll_seconds: float = 1.0,
    retry_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    stop_event: threading.Event | None = None,
    once: bool = False,
    executor_factory: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """Claim durable jobs and execute them in a bounded, process-isolated pool.

    A set stop event stops new claims and drains already submitted work.
    ``max_in_flight`` is the explicit backpressure boundary.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    capacity = max_workers if max_in_flight is None else max_in_flight
    if capacity < 1:
        raise ValueError("max_in_flight must be positive")
    registry = dict(DEFAULT_HANDLERS if handlers is None else handlers)
    db_path = str(db_path)
    models_dir = str(models_dir or (Path(db_path).parent / "models"))
    stop = stop_event or threading.Event()
    owner = f"worker:{id(stop)}"
    control: duckdb.DuckDBPyConnection | None = connect(db_path)
    init_schema(control)
    recover_stale_jobs(control)
    completed: list[dict[str, Any]] = []
    try:
        # ``spawn`` prevents child processes from inheriting an open DuckDB
        # connection through ``fork``. Tests may inject a deterministic executor;
        # production always takes this ProcessPool path.
        factory = executor_factory or (
            lambda workers: ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        )
        with factory(max_workers) as pool:
            while not stop.is_set():
                assert control is not None
                claimed = claim_job(
                    control, worker_id=owner, lease_seconds=lease_seconds
                )
                if claimed is None:
                    if once:
                        break
                    stop.wait(poll_seconds)
                    continue
                handler = registry.get(str(claimed["kind"]))
                if handler is None:
                    job = fail_claimed_job(
                        control,
                        claimed,
                        LookupError(
                            f"no handler registered for job kind {claimed['kind']!r}"
                        ),
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                    completed.append({"job": job, "ok": False})
                    if once:
                        break
                    continue

                # DuckDB permits multiple writer processes only through application
                # coordination. Release the control-process file lock before the
                # isolated worker opens its own connection. This serializes DB-bound
                # jobs (strong backpressure) while keeping all CPU/IO work out of
                # the API and scheduler processes.
                control.close()
                control = None
                future = pool.submit(
                    _execute_isolated,
                    db_path,
                    models_dir,
                    handler,
                    claimed,
                    lease_seconds,
                    heartbeat_seconds,
                )
                try:
                    result = future.result()
                except BaseException as exc:
                    control = connect(db_path)
                    job = fail_claimed_job(
                        control,
                        claimed,
                        exc,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                    completed.append({"job": job, "ok": False, "error": str(exc)})
                else:
                    control = connect(db_path)
                    job = complete_claimed_job(control, claimed, result)
                    completed.append({"job": job, "ok": True, "result": result})
                if once:
                    break
    finally:
        if control is not None:
            control.close()
    return completed


def main() -> None:
    from openenergy.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the OpenEnergy durable job worker")
    parser.add_argument("--db-path", type=Path, default=settings.db_path)
    parser.add_argument("--models-dir", type=Path, default=settings.data_dir / "models")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-in-flight", type=int)
    args = parser.parse_args()
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run_worker(
        args.db_path,
        models_dir=args.models_dir,
        max_workers=args.workers,
        max_in_flight=args.max_in_flight,
        stop_event=stop,
    )


if __name__ == "__main__":
    main()
