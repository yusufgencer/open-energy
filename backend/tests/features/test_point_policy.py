import polars as pl

from openenergy.features.point_policy import POINT_POLICIES, apply_point_policy


def _multi_raw():
    # two grid points (7=center, 8), one variable, + power_mw
    return pl.DataFrame({
        "valid_time": [0, 1],
        "wind_speed_100m__p7": [2.0, 4.0],
        "wind_speed_100m__p8": [4.0, 8.0],
        "power_mw": [1.0, 2.0],
    })


def test_policies_enumerated():
    assert set(POINT_POLICIES) >= {"single_best", "spatial_mean", "mean_plus_spread", "all_points"}


def test_single_best_uses_center_point_as_canonical():
    out = apply_point_policy(_multi_raw(), "single_best", point_ids=[7, 8])
    assert out["wind_speed_100m"].to_list() == [2.0, 4.0]  # center p7
    assert "wind_speed_100m__p7" not in out.columns  # namespaced dropped
    assert out["power_mw"].to_list() == [1.0, 2.0]  # target preserved


def test_spatial_mean_averages_points():
    out = apply_point_policy(_multi_raw(), "spatial_mean", point_ids=[7, 8])
    assert out["wind_speed_100m"].to_list() == [3.0, 6.0]  # mean(2,4), mean(4,8)


def test_mean_plus_spread_adds_std_column():
    out = apply_point_policy(_multi_raw(), "mean_plus_spread", point_ids=[7, 8])
    assert out["wind_speed_100m"].to_list() == [3.0, 6.0]
    assert abs(out["wind_speed_100m_sstd"].to_list()[0] - 1.0) < 1e-9  # std of {2,4}


def test_all_points_keeps_namespaced_and_canonical():
    out = apply_point_policy(_multi_raw(), "all_points", point_ids=[7, 8])
    assert "wind_speed_100m" in out.columns  # canonical for feature blocks
    assert "wind_speed_100m__p7" in out.columns and "wind_speed_100m__p8" in out.columns


def test_rows_and_target_stable_across_policies():
    raw = _multi_raw()
    for pol in ("single_best", "spatial_mean", "mean_plus_spread", "all_points"):
        out = apply_point_policy(raw, pol, point_ids=[7, 8])
        assert out.height == raw.height
        assert out["power_mw"].to_list() == [1.0, 2.0]
