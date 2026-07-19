import pytest

from openenergy.features.families import available_families, blocks_for_family, get_family
from openenergy.features.registry import available_blocks


def test_wind_families_reference_known_blocks():
    valid = set(available_blocks("wind"))
    for name in available_families("wind"):
        assert set(blocks_for_family(name)) <= valid


def test_solar_families_reference_known_blocks():
    valid = set(available_blocks("solar"))
    for name in available_families("solar"):
        assert set(blocks_for_family(name)) <= valid


def test_known_family_has_description():
    family = get_family("wind_physical_core")
    assert family.kind == "wind"
    assert "wind_power" in family.blocks
    assert family.description


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        get_family("nope")


def test_new_wind_families_registered():
    wind = available_families("wind")
    for name in ("wind_trajectory_physics", "wind_spatial_fusion"):
        assert name in wind
        family = get_family(name)
        assert family.kind == "wind"
        assert family.blocks
        assert family.description


def test_new_solar_family_registered():
    solar = available_families("solar")
    assert "solar_spatial_cloud" in solar
    family = get_family("solar_spatial_cloud")
    assert family.kind == "solar"
    assert family.description


def test_solar_geometry_family_registered():
    solar = available_families("solar")
    assert "solar_geometry_core" in solar
    family = get_family("solar_geometry_core")
    assert family.kind == "solar"
    assert "solar_geometry" in family.blocks
    assert "solar_geometry" in blocks_for_family("solar_geometry_core")
    assert family.description


def test_wind_trajectory_physics_bundles_expected_blocks():
    blocks = set(blocks_for_family("wind_trajectory_physics"))
    assert {"wind_power", "wind_physics_ext", "nwp_trajectory", "wind_direction"} <= blocks


def test_spatial_families_use_trajectory_block():
    assert "nwp_trajectory" in blocks_for_family("wind_spatial_fusion")
    assert "nwp_trajectory" in blocks_for_family("solar_spatial_cloud")
