import pytest

from openenergy.assets.repository import (
    Plant,
    PlantPoint,
    add_point,
    create_plant,
    get_plant,
    list_points,
)


def test_create_and_get_plant(db):
    p = create_plant(db, Plant(plant_id="wf1", name="WindFarm 1", kind="wind", capacity_mw=12.5))
    assert p.plant_id == "wf1"
    fetched = get_plant(db, "wf1")
    assert fetched is not None and fetched.capacity_mw == 12.5
    assert fetched.kind == "wind"


def test_get_missing_plant_returns_none(db):
    assert get_plant(db, "nope") is None


def test_add_and_list_points(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    pt = add_point(
        db,
        PlantPoint(point_id=None, plant_id="wf1", point_type="weather", latitude=39.9, longitude=32.8),
    )
    assert pt.point_id is not None
    pts = list_points(db, "wf1")
    assert len(pts) == 1 and pts[0].point_type == "weather"


def test_solar_point_keeps_tilt_azimuth(db):
    create_plant(db, Plant(plant_id="pv1", name="PV", kind="solar", capacity_mw=5))
    pt = add_point(
        db,
        PlantPoint(
            point_id=None, plant_id="pv1", point_type="panel_array",
            latitude=37.0, longitude=35.0, tilt=30.0, azimuth=-15.0,
        ),
    )
    stored = list_points(db, "pv1")[0]
    assert stored.tilt == 30.0 and stored.azimuth == -15.0


def test_list_plants(db):
    from openenergy.assets.repository import list_plants
    create_plant(db, Plant(plant_id="a", name="Alpha", kind="wind", capacity_mw=5))
    create_plant(db, Plant(plant_id="b", name="Beta", kind="solar", capacity_mw=8))
    plants = list_plants(db)
    assert {p.plant_id for p in plants} == {"a", "b"}
