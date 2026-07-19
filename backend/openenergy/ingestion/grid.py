"""Weather-point grid generation around a plant.

A plant's forecast quality benefits from a grid of surrounding weather points
(Andrade & Bessa 2017) rather than a single nearest point. Grid spacing is matched
to the NWP model's native resolution — Open-Meteo interpolates, so points closer
than the native cell are redundant. Points are persisted as `plant_points` rows
(point_type='weather') tagged with a shared grid_id so a point_policy can address
"the grid" as a unit. Generation is idempotent per (plant_id, grid_id, row, col).
"""
from __future__ import annotations

import math

import duckdb

from openenergy.assets.repository import PlantPoint

# Native horizontal resolution (degrees) per Open-Meteo model.
_NATIVE_SPACING: dict[str, float] = {
    "icon_eu": 0.0625,
    "icon": 0.125,
    "ecmwf_ifs025": 0.25,
    "ecmwf": 0.1,
    "gfs_global": 0.25,
    "gfs": 0.25,
    "best_match": 0.1,
}


def native_spacing(source: str) -> float:
    if source not in _NATIVE_SPACING:
        raise ValueError(f"bilinmeyen NWP kaynağı: {source}")
    return _NATIVE_SPACING[source]


def generate_grid(
    con: duckdb.DuckDBPyConnection,
    *,
    plant_id: str,
    center_lat: float,
    center_lon: float,
    source: str,
    size: int = 5,
    upwind_deg: float | None = None,
    grid_id: str | None = None,
) -> list[PlantPoint]:
    """Create (or return) an NxN weather grid centred on the plant.

    `size` is the side length (default 5 → 25 points), spacing = native
    resolution of `source`. When `upwind_deg` (meteorological bearing the wind
    comes FROM) is given, the whole grid leans half a cell toward that bearing so
    more points sit upwind. Idempotent: existing grid cells are returned, not
    duplicated.
    """
    spacing = native_spacing(source)
    half = size // 2
    gid = grid_id or f"{source}_{size}x{size}"
    clat, clon = center_lat, center_lon
    if upwind_deg is not None:
        r = math.radians(upwind_deg)
        clat += math.cos(r) * spacing * 0.5
        clon += math.sin(r) * spacing * 0.5

    points: list[PlantPoint] = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            lat = clat + i * spacing
            lon = clon + j * spacing
            existing = con.execute(
                "SELECT point_id, latitude, longitude FROM plant_points "
                "WHERE plant_id=? AND grid_id=? AND grid_row=? AND grid_col=?",
                [plant_id, gid, i, j],
            ).fetchone()
            if existing is not None:
                points.append(PlantPoint(
                    point_id=existing[0], plant_id=plant_id, point_type="weather",
                    latitude=existing[1], longitude=existing[2],
                ))
                continue
            pid = con.execute(
                "INSERT INTO plant_points "
                "(plant_id, point_type, latitude, longitude, grid_id, grid_row, grid_col) "
                "VALUES (?, 'weather', ?, ?, ?, ?, ?) RETURNING point_id",
                [plant_id, lat, lon, gid, i, j],
            ).fetchone()[0]
            points.append(PlantPoint(
                point_id=pid, plant_id=plant_id, point_type="weather",
                latitude=lat, longitude=lon,
            ))
    return points


def list_grid_points(con: duckdb.DuckDBPyConnection, plant_id: str, grid_id: str) -> list[PlantPoint]:
    rows = con.execute(
        "SELECT point_id, plant_id, point_type, latitude, longitude "
        "FROM plant_points WHERE plant_id=? AND grid_id=? ORDER BY grid_row, grid_col",
        [plant_id, grid_id],
    ).fetchall()
    return [
        PlantPoint(point_id=r[0], plant_id=r[1], point_type=r[2], latitude=r[3], longitude=r[4])
        for r in rows
    ]
