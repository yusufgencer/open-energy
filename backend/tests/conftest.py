import datetime as dt

import pytest

from openenergy.storage import connect, init_schema


@pytest.fixture
def db():
    con = connect(None)
    init_schema(con)
    yield con
    con.close()


def _constrained_valid_time(
    *,
    issue_time: dt.datetime,
    horizon_hours: int,
    valid_time: dt.datetime | None,
) -> dt.datetime:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours pozitif olmalı")
    expected = issue_time + dt.timedelta(hours=horizon_hours)
    if valid_time is not None and valid_time != expected:
        raise ValueError("valid_time, issue_time + horizon_hours olmalı")
    return expected


@pytest.fixture
def weather_row_factory():
    """Build weather rows that cannot violate issue/valid/lead-time physics."""

    def build(
        *,
        issue_time: dt.datetime,
        horizon_hours: int,
        valid_time: dt.datetime | None = None,
        point_id: int = 1,
        role: str = "previous_runs",
        model: str = "best_match",
        variable: str = "temperature_2m",
        value: float = 10.0,
    ) -> dict:
        return {
            "point_id": point_id,
            "role": role,
            "model": model,
            "issue_time": issue_time,
            "valid_time": _constrained_valid_time(
                issue_time=issue_time,
                horizon_hours=horizon_hours,
                valid_time=valid_time,
            ),
            "lead_hours": horizon_hours,
            "variable": variable,
            "value": value,
        }

    return build


@pytest.fixture
def forecast_row_factory():
    """Build forecast rows with aligned issue time, horizon, and valid time."""

    def build(
        *,
        issue_time: dt.datetime,
        horizon_hours: int,
        valid_time: dt.datetime | None = None,
        plant_id: str = "wf1",
        point_id: int = 1,
        p10: float = 1.0,
        p50: float = 2.0,
        p90: float = 3.0,
        model_id: int | None = None,
    ) -> dict:
        if not 0.0 <= p10 <= p50 <= p90:
            raise ValueError("forecast bantları p10 <= p50 <= p90 olmalı")
        return {
            "plant_id": plant_id,
            "point_id": point_id,
            "horizon_hours": horizon_hours,
            "issue_time": issue_time,
            "valid_time": _constrained_valid_time(
                issue_time=issue_time,
                horizon_hours=horizon_hours,
                valid_time=valid_time,
            ),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "model_id": model_id,
        }

    return build
