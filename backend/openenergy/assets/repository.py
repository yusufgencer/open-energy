from __future__ import annotations

from typing import Literal

import duckdb
from pydantic import BaseModel


class Plant(BaseModel):
    plant_id: str
    name: str
    kind: Literal["wind", "solar"]
    capacity_mw: float
    timezone: str = "UTC"


class PlantPoint(BaseModel):
    point_id: int | None = None
    plant_id: str
    point_type: Literal["weather", "turbine", "panel_array"]
    latitude: float
    longitude: float
    hub_height_m: float | None = None
    tilt: float | None = None
    azimuth: float | None = None


def create_plant(con: duckdb.DuckDBPyConnection, plant: Plant) -> Plant:
    con.execute(
        "INSERT INTO plants (plant_id, name, kind, capacity_mw, timezone) VALUES (?,?,?,?,?)",
        [plant.plant_id, plant.name, plant.kind, plant.capacity_mw, plant.timezone],
    )
    return plant


def get_plant(con: duckdb.DuckDBPyConnection, plant_id: str) -> Plant | None:
    row = con.execute(
        "SELECT plant_id, name, kind, capacity_mw, timezone FROM plants WHERE plant_id=?",
        [plant_id],
    ).fetchone()
    if row is None:
        return None
    return Plant(plant_id=row[0], name=row[1], kind=row[2], capacity_mw=row[3], timezone=row[4])


def list_plants(con: duckdb.DuckDBPyConnection) -> list[Plant]:
    rows = con.execute(
        "SELECT plant_id, name, kind, capacity_mw, timezone FROM plants ORDER BY name"
    ).fetchall()
    return [
        Plant(plant_id=r[0], name=r[1], kind=r[2], capacity_mw=r[3], timezone=r[4])
        for r in rows
    ]


def add_point(con: duckdb.DuckDBPyConnection, point: PlantPoint) -> PlantPoint:
    new_id = con.execute(
        """
        INSERT INTO plant_points
            (plant_id, point_type, latitude, longitude, hub_height_m, tilt, azimuth)
        VALUES (?,?,?,?,?,?,?)
        RETURNING point_id
        """,
        [
            point.plant_id, point.point_type, point.latitude, point.longitude,
            point.hub_height_m, point.tilt, point.azimuth,
        ],
    ).fetchone()[0]
    return point.model_copy(update={"point_id": new_id})


def list_points(con: duckdb.DuckDBPyConnection, plant_id: str) -> list[PlantPoint]:
    rows = con.execute(
        """
        SELECT point_id, plant_id, point_type, latitude, longitude, hub_height_m, tilt, azimuth
        FROM plant_points WHERE plant_id=? ORDER BY point_id
        """,
        [plant_id],
    ).fetchall()
    return [
        PlantPoint(
            point_id=r[0], plant_id=r[1], point_type=r[2], latitude=r[3], longitude=r[4],
            hub_height_m=r[5], tilt=r[6], azimuth=r[7],
        )
        for r in rows
    ]
