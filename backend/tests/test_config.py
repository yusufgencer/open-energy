from pathlib import Path

from openenergy.config import Settings


def test_db_path_derived_from_data_dir(tmp_path: Path):
    s = Settings(data_dir=tmp_path)
    assert s.db_path == tmp_path / "openenergy.duckdb"


def test_ensure_dirs_creates_data_dir(tmp_path: Path):
    target = tmp_path / "nested"
    s = Settings(data_dir=target)
    s.ensure_dirs()
    assert target.is_dir()


def test_single_runs_disabled_by_default():
    # Open-Meteo Single Runs API is paid; it MUST never be enabled by default.
    s = Settings()
    assert s.openmeteo_single_runs_enabled is False


def test_single_runs_enable_via_env(monkeypatch):
    monkeypatch.setenv("OPENENERGY_OPENMETEO_SINGLE_RUNS_ENABLED", "1")
    s = Settings()
    assert s.openmeteo_single_runs_enabled is True
