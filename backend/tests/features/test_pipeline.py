import polars as pl
from openenergy.features import build_features, default_config, FeatureConfig


def _wind_df():
    return pl.DataFrame({
        "valid_time": [None, None], "power_mw": [1.0, 2.0],
        "wind_speed_100m": [3.0, 4.0], "wind_speed_10m": [1.0, 2.0],
        "surface_pressure": [1013.0, 1010.0], "temperature_2m": [15.0, 20.0],
    })


def test_build_features_applies_selected_blocks():
    cfg = FeatureConfig(blocks=["wind_power", "air_density"])
    out = build_features(_wind_df(), "wind", cfg)
    assert "wind_speed_cube" in out.columns and "rho_v3" in out.columns
    assert "wind_shear" not in out.columns          # seçilmedi
    assert out["power_mw"].to_list() == [1.0, 2.0]   # hedef korunur


def test_default_config_runs_all_wind_blocks():
    out = build_features(_wind_df(), "wind", default_config("wind"))
    assert "wind_speed_cube" in out.columns


def test_wrong_kind_block_raises():
    import pytest
    cfg = FeatureConfig(blocks=["clear_sky_index"])
    with pytest.raises(ValueError):
        build_features(_wind_df(), "wind", cfg)


def test_feature_config_rejects_duplicate_blocks():
    import pytest
    with pytest.raises(ValueError):
        FeatureConfig(blocks=["wind_power", "wind_power"])
