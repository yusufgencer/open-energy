from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenergy.assets.repository import (
    Plant, PlantPoint, add_point, create_plant, get_plant, list_plants, list_points,
)
from openenergy.config import Settings, get_settings
from openenergy.experiments.persistence import best_trials
from openenergy.experiments.scenario_memory import (
    plant_strategy_profile,
    policy_leaderboard,
    scenario_leaderboard,
)
from openenergy.forecasting.backtest import backtest_champion
from openenergy.ingestion.grid import generate_grid, list_grid_points
from openenergy.ingestion.production import load_production_csv, write_production
from openenergy.integrations import epias_store
from openenergy.jobs.runner import (
    create_job, enqueue_emergency_retrain, get_job, list_jobs,
)
from openenergy.retraining.drift import check_drift, drift_status
from openenergy.providers.epias.client import EpiasAuthError
from openenergy.providers.epias.generation import list_powerplants
from openenergy.storage import connect, init_schema


# ── Request bodies ─────────────────────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    plant_id: str
    point_id: int
    horizons: list[int]
    kind: str
    capacity_mw: float
    nwp_sources: list[str] = ["icon"]
    n_trials: int = 10
    top_k: int = 10
    search_mode: str = "strategy"
    preset: str = "balanced"
    # B4-t2: optional policy subsets. point_ids selects the grid subset the
    # point_policy search runs over; source pins the NWP source_policy axis.
    # Both default to None → single-point path (all applicable policies).
    point_ids: list[int] | None = None
    source: str | None = None


class RetrainRequest(BaseModel):
    # E1-t2: scheduled/manual base retrain — re-run champion-challenger for a plant
    # across horizons in-process (DuckDB single-writer). Mirrors ExperimentRequest
    # but drives openenergy.retraining.champion.retrain (significance-gated promote).
    plant_id: str
    point_id: int
    horizons: list[int]
    kind: str
    capacity_mw: float
    nwp_sources: list[str] = ["icon"]
    n_trials: int = 10
    n_splits: int = 4
    embargo: int = 24
    seed: int = 42


class PostprocessorFitRequest(BaseModel):
    # E1-t3: refit the daily LASSO polynomial quantile post-processor for a plant's
    # horizons from recent (forecast, actual) pairs. Cheap drift/bias correction
    # between monthly base retrains; identity when there is no recent data.
    horizons: list[int]
    degree: int = 2
    alpha: float = 0.001
    min_samples: int = 12
    lookback_days: int = 45


class ReforecastRequest(BaseModel):
    # F1-t4: reforecast on NWP-run arrival, blending the fresh run with the prior
    # (w_new/1-w_new) to cut jumpiness. Fires once per issue_time (NWP init).
    point_id: int
    horizon_hours: int
    kind: str
    capacity_mw: float | None = None  # defaults to the plant's registered capacity
    issue_time: datetime | None = None  # defaults to now (UTC)
    w_new: float = 0.7


class ForecastRequest(BaseModel):
    point_id: int
    horizons: list[int]
    kind: str
    capacity_mw: float | None = None
    issue_time: datetime | None = None
    models: list[str] | None = None


class ForecastBackfillRequest(BaseModel):
    point_id: int
    horizons: list[int]
    start_time: datetime | None = None
    end_time: datetime | None = None
    lookback_days: int = 92


class DriftCheckRequest(BaseModel):
    # E1-t4: run the drift monitor for a plant's horizons against the champion's
    # proven skill. A persistent breach enqueues an emergency retrain (once, with
    # hysteresis) using the same champion-challenger retrain the monthly cycle runs.
    point_id: int
    horizons: list[int]
    kind: str
    capacity_mw: float | None = None  # defaults to the plant's registered capacity
    nwp_sources: list[str] = ["icon"]
    n_trials: int = 10
    n_splits: int = 4
    embargo: int = 24
    seed: int = 42
    lookback_days: int = 30
    min_samples: int = 12
    breach_ratio: float = 1.5
    recover_ratio: float = 1.2
    min_consecutive: int = 2
    psi_threshold: float = 0.25
    ks_alpha: float = 0.05
    freshness_hours: float = 36.0


class WeatherGridRequest(BaseModel):
    source: str              # NWP kaynağı (icon_eu, ecmwf_ifs025, gfs_global, ...)
    size: int = 5            # NxN grid kenar uzunluğu (varsayılan 5 → 25 nokta)
    upwind_deg: float | None = None  # rüzgârın geldiği yön (meteo bearing); grid o yöne kayar


class WeatherIngestRequest(BaseModel):
    # Tek nokta (legacy) VEYA bir grid (çok nokta/çok kaynak) doldurulur.
    point_id: int | None = None
    grid_id: str | None = None
    sources: list[str] | None = None  # grid modu: açık NWP model listesi (source_policy ekseni)
    past_days: int = 92      # Open-Meteo previous-runs: en fazla ~92 gün geçmiş
    # F1-t1: previous_day1..7 → lead 24..168 → horizon 24/48/72/96/120/144/168
    # eşleşir. Uzatılan horizonlar (≥72h) yalnızca skill kanıtlanınca champion olur.
    previous_days: int = 7


class EpiasConnectRequest(BaseModel):
    username: str
    password: str
    persist: bool = True


class EpiasIngestRequest(BaseModel):
    power_plant_id: int | None = None  # verilmezse plant_name ile çözülür
    plant_name: str | None = None      # ör. "BARES" / "BALIKESIR"
    start_date: str | None = None      # YYYY-MM-DD (boşsa bugünden 92 gün öncesi)
    end_date: str | None = None        # YYYY-MM-DD (boşsa bugün)
    field: str = "total"               # "total" (santral toplamı) veya "wind"


def _representative_point(
    con: duckdb.DuckDBPyConnection, plant_id: str
) -> tuple[int, float, float] | None:
    """Return (point_id, lat, lon) of the plant's representative point.

    The representative point is the plant's own (non-grid) point — grid cells
    carry a grid_id, so we pick the earliest point with grid_id IS NULL. A grid
    is built around these coordinates.
    """
    row = con.execute(
        "SELECT point_id, latitude, longitude FROM plant_points "
        "WHERE plant_id=? AND grid_id IS NULL ORDER BY point_id LIMIT 1",
        [plant_id],
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), float(row[1]), float(row[2])


# ── App factory ────────────────────────────────────────────────────────────────

def create_app(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI app. Inject `con` for testing (in-memory); default uses settings db.

    models_dir is computed once at app creation time (not per request) to avoid temp-dir leaks:
    - Production (con=None): uses settings.data_dir/models (stable across restarts).
    - Testing (con injected): uses a single mkdtemp() per app instance.
    """
    settings = settings or get_settings()
    if con is None:
        # Kalıcı dosya-tabanlı DuckDB: restart'ta veri korunur, ve ingestion
        # script'leri ile API aynı DB'yi paylaşır (gerçek santral testi için şart).
        settings.ensure_dirs()
        con = connect(settings.db_path)
        init_schema(con)  # IF NOT EXISTS — mevcut DB'de idempotent
        _models_dir: Path = settings.data_dir / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
    else:
        _models_dir = Path(tempfile.mkdtemp())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = None
        if settings.scheduler_enabled:
            # Imported lazily so normal API/test startup stays free of scheduler
            # side effects. The producer receives a fresh cursor on every tick.
            from openenergy.jobs.scheduler import build_scheduler

            scheduler = build_scheduler(lambda: con.cursor(), settings)
            scheduler.start()
            app.state.scheduler = scheduler
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(title="OpenEnergy API", lifespan=lifespan)

    def db() -> duckdb.DuckDBPyConnection:
        # DuckDB bağlantısı thread-safe değil; FastAPI sync endpoint'leri threadpool'da
        # çalışır. Her çağrıya aynı in-memory DB'ye bağlı izole bir cursor ver → eşzamanlı
        # isteklerde result-state çakışması (yanlış 404/veri) olmaz.
        return con.cursor()

    def queued(job_id: int) -> dict[str, Any]:
        return {"job_id": job_id, "status": "pending"}

    # Lokal frontend (Next.js dev) tarayıcıdan API'ye erişebilsin diye CORS.
    _origins = os.environ.get("OPENENERGY_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/fleet/health")
    def fleet_health() -> list[dict]:
        """Compact, query-only health matrix for every registered plant."""
        rows = db().execute(
            """
            SELECT p.plant_id, p.name, p.kind, p.capacity_mw,
                   (SELECT max(f.issue_time) FROM forecasts f
                    WHERE f.plant_id=p.plant_id) AS latest_forecast_at,
                   (SELECT max(pr.ts) FROM production pr
                    WHERE pr.plant_id=p.plant_id) AS latest_production_at,
                   CASE
                     WHEN EXISTS (
                       SELECT 1 FROM drift_events de
                       WHERE de.plant_id=p.plant_id
                         AND de.detector_status IN ('stale','missing','breach','drift')
                     ) THEN 'critical'
                     WHEN EXISTS (
                       SELECT 1 FROM drift_state ds
                       WHERE ds.plant_id=p.plant_id
                         AND (ds.fired OR ds.consecutive_breaches > 0)
                     ) THEN 'critical'
                     WHEN EXISTS (
                       SELECT 1 FROM drift_events de
                       WHERE de.plant_id=p.plant_id
                         AND de.detector_status IN ('warning','insufficient_data')
                     ) THEN 'warning'
                     WHEN EXISTS (
                       SELECT 1 FROM drift_events de
                       WHERE de.plant_id=p.plant_id AND de.detector_status='ok'
                     ) THEN 'healthy'
                     ELSE 'unknown'
                   END AS drift_status,
                   (SELECT count(*) FROM jobs j
                    WHERE json_extract_string(j.payload, '$.plant_id')=p.plant_id
                      AND j.status IN ('pending','running','failed')) AS active_jobs
            FROM plants p
            ORDER BY p.name, p.plant_id
            """
        ).fetchall()
        payload = []
        for row in rows:
            drift = str(row[6])
            latest_forecast = row[4]
            latest_production = row[5]
            if drift == "critical":
                status = "critical"
            elif drift == "warning":
                status = "warning"
            elif drift == "healthy" and latest_forecast is not None and latest_production is not None:
                status = "healthy"
            else:
                status = "unknown"
            payload.append({
                "plant_id": row[0], "name": row[1], "kind": row[2],
                "capacity_mw": float(row[3]), "status": status,
                "latest_forecast_at": (
                    None if latest_forecast is None else str(latest_forecast)
                ),
                "latest_production_at": (
                    None if latest_production is None else str(latest_production)
                ),
                "drift_status": drift, "active_jobs": int(row[7]),
            })
        return payload

    # ── Plants ──────────────────────────────────────────────────────────────────

    @app.post("/plants")
    def create_plant_endpoint(plant: Plant) -> dict:
        created = create_plant(db(), plant)
        return created.model_dump()

    @app.get("/plants")
    def list_plants_endpoint() -> list[dict]:
        return [p.model_dump() for p in list_plants(db())]

    @app.get("/plants/{plant_id}")
    def get_plant_endpoint(plant_id: str) -> dict:
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        points = list_points(c, plant_id)
        return {"plant": plant.model_dump(), "points": [p.model_dump() for p in points]}

    @app.post("/plants/{plant_id}/points")
    def add_point_endpoint(plant_id: str, point: PlantPoint) -> dict:
        created = add_point(db(), point)
        return created.model_dump()

    @app.post("/plants/{plant_id}/production")
    async def upload_production(plant_id: str, file: UploadFile = File(...)) -> dict:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            frame = load_production_csv(tmp_path)
            rows = write_production(db(), plant_id, frame)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return {"rows": rows}

    # ── EPİAŞ entegrasyonu ──────────────────────────────────────────────────────

    @app.get("/integrations/epias/status")
    def epias_status() -> dict:
        return epias_store.status()

    @app.post("/integrations/epias/connect")
    def epias_connect(req: EpiasConnectRequest) -> dict:
        # Önce TGT alarak credential'ı doğrula; başarısızsa kaydetme.
        try:
            epias_store.test_connection(req.username, req.password)
        except EpiasAuthError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:  # ağ/HTTP hataları
            raise HTTPException(status_code=502, detail=f"EPİAŞ'a bağlanılamadı: {e}")
        epias_store.set_credentials(req.username, req.password, persist=req.persist)
        return epias_store.status()

    @app.post("/integrations/epias/disconnect")
    def epias_disconnect() -> dict:
        epias_store.clear_credentials()
        return epias_store.status()

    @app.get("/integrations/epias/powerplants")
    def epias_powerplants(q: str | None = None) -> list[dict]:
        try:
            client = epias_store.get_client()
        except EpiasAuthError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            items = list_powerplants(client)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"EPİAŞ santral listesi alınamadı: {e}")
        if q:
            key = q.upper()
            items = [it for it in items
                     if key in str(it.get("name") or it.get("shortName") or "").upper()]
        # UI için sade alanlar
        return [
            {"id": it.get("id"), "name": it.get("name") or it.get("shortName"),
             "eic": it.get("eic")}
            for it in items[:50]
        ]

    # ── Weather grid ────────────────────────────────────────────────────────────

    @app.post("/plants/{plant_id}/weather-grid")
    def create_weather_grid_endpoint(plant_id: str, req: WeatherGridRequest) -> dict:
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        rep = _representative_point(c, plant_id)
        if rep is None:
            raise HTTPException(
                status_code=400,
                detail="Plant has no representative point (add a plant point first)",
            )
        _, lat, lon = rep
        try:
            points = generate_grid(
                c,
                plant_id=plant_id,
                center_lat=lat,
                center_lon=lon,
                source=req.source,
                size=req.size,
                upwind_deg=req.upwind_deg,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        grid_id = f"{req.source}_{req.size}x{req.size}"
        return {
            "grid_id": grid_id,
            "source": req.source,
            "size": req.size,
            "count": len(points),
            "points": [p.model_dump() for p in points],
        }

    # ── Weather ingestion ───────────────────────────────────────────────────────

    @app.post("/plants/{plant_id}/ingest-weather")
    def ingest_weather_endpoint(plant_id: str, req: WeatherIngestRequest) -> dict:
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        kind = plant.kind

        # Grid modu: çok nokta / çok kaynak previous-runs ingest.
        if req.grid_id is not None:
            points = list_grid_points(c, plant_id, req.grid_id)
            if not points:
                raise HTTPException(status_code=404, detail="Grid not found")
            job_id = create_job(
                db(), "ingest_weather_grid",
                f"plant={plant_id} grid={req.grid_id} points={len(points)}",
                payload={
                    "plant_id": plant_id,
                    "grid_id": req.grid_id,
                    "kind": kind,
                    "sources": req.sources,
                    "past_days": req.past_days,
                    "previous_days": req.previous_days,
                },
            )
            return queued(job_id)

        # Legacy tek-nokta modu (byte-identical davranış).
        if req.point_id is None:
            raise HTTPException(status_code=400, detail="point_id veya grid_id gerekli")
        point = next((p for p in list_points(c, plant_id) if p.point_id == req.point_id), None)
        if point is None:
            raise HTTPException(status_code=404, detail="Point not found")

        job_id = create_job(
            db(),
            "ingest_weather",
            f"plant={plant_id} point={req.point_id}",
            payload={
                "plant_id": plant_id,
                "point_id": req.point_id,
                "kind": kind,
                "past_days": req.past_days,
                "previous_days": req.previous_days,
            },
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/ingest-epias")
    def ingest_epias_endpoint(plant_id: str, req: EpiasIngestRequest) -> dict:
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        if req.power_plant_id is None and not req.plant_name:
            raise HTTPException(status_code=400, detail="power_plant_id veya plant_name gerekli")

        end = date.fromisoformat(req.end_date) if req.end_date else date.today()
        start = date.fromisoformat(req.start_date) if req.start_date else end - timedelta(days=92)

        try:
            epias_store.get_client()
        except EpiasAuthError as e:
            raise HTTPException(status_code=400, detail=str(e))

        job_id = create_job(
            db(),
            "ingest_epias",
            f"plant={plant_id}",
            payload={
                "plant_id": plant_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "power_plant_id": req.power_plant_id,
                "plant_name": req.plant_name,
                "field": req.field,
            },
        )
        return queued(job_id)

    # ── Experiments ─────────────────────────────────────────────────────────────

    @app.post("/experiments")
    def start_experiment(req: ExperimentRequest) -> dict:
        job_id = create_job(
            db(),
            "experiment",
            f"plant={req.plant_id}",
            payload=req.model_dump(mode="json"),
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/retrain")
    def start_retrain(plant_id: str, req: RetrainRequest) -> dict:
        # E1-t2: periodic (monthly) full retrain — re-challenge the champion against
        # the growing pool per horizon, in-process. One call re-runs the whole
        # search-and-prove loop for a plant across all requested horizons.
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")

        payload = req.model_dump(mode="json")
        payload["plant_id"] = plant_id
        job_id = create_job(
            db(),
            "retrain",
            f"plant={plant_id} horizons={req.horizons}",
            payload=payload,
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/reforecast")
    def start_reforecast(plant_id: str, req: ReforecastRequest) -> dict:
        # F1-t4: reforecast on NWP-run arrival + time-lagged blend with the prior run.
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        cap = req.capacity_mw if req.capacity_mw is not None else plant.capacity_mw
        issue = req.issue_time or datetime.utcnow()
        job_id = create_job(
            db(),
            "reforecast",
            f"plant={plant_id} h={req.horizon_hours}",
            payload={
                **req.model_dump(mode="json"),
                "plant_id": plant_id,
                "capacity_mw": cap,
                "issue_time": issue.isoformat(),
            },
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/forecast")
    def start_forecast(plant_id: str, req: ForecastRequest) -> dict:
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        point = next((p for p in list_points(c, plant_id) if p.point_id == req.point_id), None)
        if point is None:
            raise HTTPException(status_code=404, detail="Plant point not found")
        if req.kind != plant.kind:
            raise HTTPException(status_code=400, detail="Forecast kind does not match plant")
        cap = req.capacity_mw if req.capacity_mw is not None else plant.capacity_mw
        issue = req.issue_time or datetime.utcnow()
        job_id = create_job(
            db(),
            "forecast",
            f"plant={plant_id} horizons={req.horizons}",
            payload={
                **req.model_dump(mode="json"),
                "plant_id": plant_id,
                "capacity_mw": cap,
                "issue_time": issue.isoformat(),
            },
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/forecasts/backfill")
    def start_forecast_backfill(
        plant_id: str, req: ForecastBackfillRequest
    ) -> dict:
        """Backfill realised previous-runs targets through the deployed champion."""
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        point = next((p for p in list_points(c, plant_id) if p.point_id == req.point_id), None)
        if point is None:
            raise HTTPException(status_code=404, detail="Plant point not found")

        job_id = create_job(
            db(),
            "forecast_backfill",
            f"plant={plant_id} horizons={req.horizons}",
            payload={
                **req.model_dump(mode="json"),
                "plant_id": plant_id,
                "capacity_mw": plant.capacity_mw,
                "kind": plant.kind,
            },
        )
        return queued(job_id)

    @app.post("/plants/{plant_id}/postprocessor/fit")
    def start_postprocessor_fit(
        plant_id: str, req: PostprocessorFitRequest
    ) -> dict:
        # E1-t3: refit the per-plant+horizon LASSO polynomial post-processor from
        # recent (forecast, actual) pairs, in-process (DuckDB single-writer). The
        # fitted coefficients are applied on top of champion forecasts at serving.
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")

        job_id = create_job(
            db(),
            "postprocessor_fit",
            f"plant={plant_id} horizons={req.horizons}",
            payload={**req.model_dump(mode="json"), "plant_id": plant_id},
        )
        return queued(job_id)

    @app.get("/plants/{plant_id}/drift")
    def get_drift_status(plant_id: str) -> list[dict]:
        # E1-t4: status surface — persisted per-horizon drift state (breaching flag,
        # recent vs champion nRMSE, last emergency-retrain job id).
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        return drift_status(c, plant_id)

    @app.post("/plants/{plant_id}/drift/check")
    def start_drift_check(plant_id: str, req: DriftCheckRequest) -> dict:
        # E1-t4: evaluate rolling recent error vs the champion's skill per horizon.
        # A persistent breach enqueues an emergency retrain (once, hysteresis) that
        # re-runs champion-challenger — sustained skill loss self-heals.
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        capacity_mw = req.capacity_mw if req.capacity_mw is not None else plant.capacity_mw
        def _make_enqueue(horizon: int):
            def _enqueue(p_id: str, h: int) -> int:
                # Record the queued emergency retrain immediately (observable even
                # before it completes), then run the same significance-gated
                # champion-challenger the monthly cycle uses, for this horizon only.
                return enqueue_emergency_retrain(
                    db(),
                    p_id,
                    h,
                    payload={
                        "plant_id": p_id,
                        "point_id": req.point_id,
                        "horizons": [h],
                        "capacity_mw": capacity_mw,
                        "kind": req.kind,
                        "nwp_sources": req.nwp_sources,
                        "n_trials": req.n_trials,
                        "n_splits": req.n_splits,
                        "embargo": req.embargo,
                        "seed": req.seed,
                    },
                )

            return _enqueue

        results = []
        for h in req.horizons:
            results.append(
                check_drift(
                    c, plant_id=plant_id, horizon_hours=h, capacity_mw=capacity_mw,
                    lookback_days=req.lookback_days, min_samples=req.min_samples,
                    breach_ratio=req.breach_ratio, recover_ratio=req.recover_ratio,
                    min_consecutive=req.min_consecutive, enqueue=_make_enqueue(h),
                    point_id=req.point_id, nwp_sources=req.nwp_sources,
                    psi_threshold=req.psi_threshold, ks_alpha=req.ks_alpha,
                    freshness_hours=req.freshness_hours,
                )
            )
        return {"results": results}

    @app.get("/jobs")
    def list_jobs_endpoint(limit: int = 50) -> list[dict]:
        try:
            return list_jobs(db(), limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/jobs/{job_id}")
    def get_job_endpoint(job_id: int) -> dict:
        j = get_job(db(), job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return j

    def _winning_policies(
        c: duckdb.DuckDBPyConnection, experiment_id: int
    ) -> dict[int, dict]:
        """Per-horizon champion scenario_run for this experiment, keyed by horizon.

        scenario_runs is the authoritative record of the winning policy combo
        (B3-t1). We surface point_policy/source_policy plus the champion skill.
        """
        result = c.execute(
            """
            SELECT horizon_hours, point_policy, source_policy, strategy_family,
                   nrmse, skill_score, evaluation_grade, evaluation_n_days,
                   evaluation_origins, evaluation_seasons
            FROM scenario_runs
            WHERE experiment_id=? AND is_champion=true
            """,
            [experiment_id],
        )
        cols = [d[0] for d in result.description]
        return {
            int(row[0]): dict(zip(cols, row)) for row in result.fetchall()
        }

    @app.get("/experiments/{experiment_id}/results")
    def get_experiment_results(experiment_id: int) -> list[dict]:
        c = db()
        rows = best_trials(c, experiment_id)
        winners = _winning_policies(c, experiment_id)
        # B4-t2: expose the winning point_policy/source_policy alongside each
        # per-horizon best trial so the winner is explainable via API alone.
        for row in rows:
            win = winners.get(int(row["horizon_hours"])) if "horizon_hours" in row else None
            row["point_policy"] = win["point_policy"] if win else None
            row["source_policy"] = win["source_policy"] if win else None
        return rows

    @app.get("/experiments/{experiment_id}/champion")
    def get_experiment_champion(experiment_id: int) -> list[dict]:
        # B4-t2: per-horizon champion payload — winning policies + skill.
        c = db()
        winners = _winning_policies(c, experiment_id)
        payload: list[dict] = []
        for horizon in sorted(winners):
            win = winners[horizon]
            skill = win["skill_score"]
            payload.append(
                {
                    "horizon_hours": horizon,
                    "point_policy": win["point_policy"],
                    "source_policy": win["source_policy"],
                    "strategy_family": win["strategy_family"],
                    "nrmse": win["nrmse"],
                    "skill_score": skill,
                    "evaluation_grade": win["evaluation_grade"],
                    "evaluation_n_days": win["evaluation_n_days"],
                    "evaluation_origins": win["evaluation_origins"],
                    "evaluation_seasons": win["evaluation_seasons"],
                    # Skill is measured against the strongest (min-error) baseline;
                    # expose it as a per-baseline map so the UI can render
                    # "skill +X vs <baseline>". Only the best-baseline skill is
                    # persisted today (runner.skill_score), keyed "best".
                    "baseline_skills": {"best": skill},
                }
            )
        return payload

    # ── Scenario Memory ────────────────────────────────────────────────────────

    @app.get("/scenario-memory/leaderboard")
    def get_scenario_leaderboard(kind: str | None = None, horizon: int | None = None) -> list[dict]:
        return scenario_leaderboard(db(), asset_kind=kind, horizon_hours=horizon)

    @app.get("/scenario-memory/policy-leaderboard")
    def get_policy_leaderboard(kind: str | None = None, horizon: int | None = None) -> list[dict]:
        return policy_leaderboard(db(), asset_kind=kind, horizon_hours=horizon)

    @app.get("/plants/{plant_id}/policy-leaderboard")
    def get_plant_policy_leaderboard(
        plant_id: str, kind: str | None = None, horizon: int | None = None
    ) -> list[dict]:
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        return policy_leaderboard(c, asset_kind=kind, horizon_hours=horizon, plant_id=plant_id)

    @app.get("/plants/{plant_id}/strategy-profile")
    def get_strategy_profile(plant_id: str) -> list[dict]:
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        return plant_strategy_profile(c, plant_id)

    @app.get("/plants/{plant_id}/scenario-runs")
    def get_plant_scenario_runs(plant_id: str) -> list[dict]:
        c = db()
        if get_plant(c, plant_id) is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        result = c.execute(
            """
            SELECT scenario_run_id, experiment_id, plant_id, asset_kind, horizon_hours,
                   strategy_trial_id, strategy_family, feature_signature, model_signature,
                   train_policy, point_policy, ensemble_policy, quantile_policy,
                   nrmse, nmae, bias, pinball, coverage, crps, peak_bias, peak_mae,
                   skill_score, evaluation_grade, evaluation_n_days,
                   evaluation_origins, evaluation_seasons,
                   runtime_seconds, is_champion, created_at
            FROM scenario_runs
            WHERE plant_id=?
            ORDER BY created_at DESC, scenario_run_id DESC
            """,
            [plant_id],
        )
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    # ── Forecasts ───────────────────────────────────────────────────────────────

    @app.get("/plants/{plant_id}/backtest")
    def get_backtest(plant_id: str, horizon: int, point_id: int | None = None) -> list[dict]:
        # Champion's held-out test-window predictions vs actual (MW) as a time series.
        c = db()
        plant = get_plant(c, plant_id)
        if plant is None:
            raise HTTPException(status_code=404, detail="Plant not found")
        return backtest_champion(
            c, plant_id=plant_id, horizon_hours=horizon,
            capacity_mw=plant.capacity_mw, kind=plant.kind, point_id=point_id,
        )

    @app.get("/plants/{plant_id}/forecasts")
    def get_forecasts(plant_id: str, horizon: int | None = None) -> list[dict]:
        query = (
            "SELECT plant_id, point_id, horizon_hours, issue_time, valid_time, "
            "p10, p50, p90, model_id, strategy_trial_id FROM forecasts WHERE plant_id=?"
        )
        params: list[Any] = [plant_id]
        if horizon is not None:
            query += " AND horizon_hours=?"
            params.append(horizon)
            query += (
                " AND issue_time=(SELECT max(f2.issue_time) FROM forecasts f2 "
                "WHERE f2.plant_id=? AND f2.horizon_hours=?)"
            )
            params.extend([plant_id, horizon])
        else:
            query += (
                " AND issue_time=(SELECT max(f2.issue_time) FROM forecasts f2 "
                "WHERE f2.plant_id=? AND f2.horizon_hours=forecasts.horizon_hours)"
            )
            params.append(plant_id)
        query += " ORDER BY valid_time"
        rows = db().execute(query, params).fetchall()
        cols = ["plant_id", "point_id", "horizon_hours", "issue_time", "valid_time",
                "p10", "p50", "p90", "model_id", "strategy_trial_id"]
        return [dict(zip(cols, r)) for r in rows]

    return app
