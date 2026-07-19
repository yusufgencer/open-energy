from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENENERGY_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=_DEFAULT_DATA_DIR)
    openmeteo_api_key: str | None = None
    # Open-Meteo Single Runs API (run= param, arbitrary leak-free leads) is a paid
    # endpoint. It MUST stay OFF by default; enabling requires this flag AND a key.
    openmeteo_single_runs_enabled: bool = False
    # Raw weather retention is independent of paid-provider enablement.  Long
    # training/backfill roles keep substantially wider windows than ephemeral
    # serving products, and every default can be overridden through settings.
    weather_retention_enabled: bool = True
    weather_retention_forecast_days: int = Field(default=30, gt=0)
    weather_retention_archive_days: int = Field(default=1825, gt=0)
    weather_retention_previous_runs_days: int = Field(default=730, gt=0)
    weather_retention_ensemble_days: int = Field(default=30, gt=0)
    weather_retention_single_runs_days: int = Field(default=180, gt=0)
    # The API may host the lightweight schedule producer, but it is deliberately
    # opt-in so tests and multi-process API deployments never create surprise
    # scheduler threads. The durable worker remains a separate process.
    scheduler_enabled: bool = False
    scheduler_timezone: str = "UTC"
    scheduler_monthly_retrain_enabled: bool = True
    scheduler_drift_enabled: bool = True
    scheduler_serving_enabled: bool = True
    scheduler_reforecast_enabled: bool = True
    scheduler_horizons: list[int] = Field(default_factory=lambda: [24, 48, 72])
    scheduler_nwp_sources: list[str] = Field(default_factory=lambda: ["icon"])
    scheduler_retrain_day: int = Field(default=1, ge=1, le=31)
    scheduler_retrain_hour: int = Field(default=2, ge=0, le=23)
    scheduler_drift_hour: int = Field(default=4, ge=0, le=23)
    scheduler_serving_minutes: int = Field(default=60, gt=0)
    scheduler_reforecast_minutes: int = Field(default=360, gt=0)
    # EPİAŞ credential'ları OPENENERGY_ prefix'i OLMADAN okunur (EPIAS_USERNAME/PASSWORD),
    # böylece backend/.env'e direkt yazılabilir (.env gitignore'lu, şifre repoda durmaz).
    epias_username: str | None = Field(
        default=None, validation_alias=AliasChoices("EPIAS_USERNAME"))
    epias_password: str | None = Field(
        default=None, validation_alias=AliasChoices("EPIAS_PASSWORD"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "openenergy.duckdb"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
