from openenergy.assets.repository import Plant, create_plant
from openenergy.ingestion import grid


def test_native_spacing_known_sources():
    assert grid.native_spacing("icon_eu") == 0.0625
    assert grid.native_spacing("gfs_global") == 0.25


def test_generate_grid_creates_nxn_native_spacing(db):
    create_plant(db, Plant(plant_id="p1", name="P", kind="wind", capacity_mw=10))
    pts = grid.generate_grid(db, plant_id="p1", center_lat=39.0, center_lon=28.0,
                             source="icon_eu", size=3)
    assert len(pts) == 9
    lats = sorted({round(p.latitude, 6) for p in pts})
    assert len(lats) == 3
    assert abs((lats[1] - lats[0]) - 0.0625) < 1e-9  # native icon_eu spacing
    # center point coincides with the plant centre
    assert any(abs(p.latitude - 39.0) < 1e-9 and abs(p.longitude - 28.0) < 1e-9 for p in pts)
    cnt = db.execute(
        "SELECT count(*) FROM plant_points WHERE plant_id='p1' AND point_type='weather'"
    ).fetchone()[0]
    assert cnt == 9


def test_generate_grid_is_idempotent(db):
    create_plant(db, Plant(plant_id="p2", name="P", kind="wind", capacity_mw=10))
    grid.generate_grid(db, plant_id="p2", center_lat=39, center_lon=28, source="icon_eu", size=3)
    grid.generate_grid(db, plant_id="p2", center_lat=39, center_lon=28, source="icon_eu", size=3)
    cnt = db.execute("SELECT count(*) FROM plant_points WHERE plant_id='p2'").fetchone()[0]
    assert cnt == 9  # regeneration must not duplicate


def test_generate_grid_upwind_shifts_centroid(db):
    create_plant(db, Plant(plant_id="p3", name="P", kind="wind", capacity_mw=10))
    pts = grid.generate_grid(db, plant_id="p3", center_lat=39.0, center_lon=28.0,
                             source="icon_eu", size=3, upwind_deg=0.0)
    mean_lat = sum(p.latitude for p in pts) / len(pts)
    assert mean_lat > 39.0  # upwind_deg=0 (north) leans the grid north
