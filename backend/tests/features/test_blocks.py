import numpy as np
import polars as pl

from openenergy.features import blocks


def test_nwp_trajectory_lag_lead_match_shifts():
    df = pl.DataFrame({"wind_speed_100m": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = blocks.nwp_trajectory(df)
    assert out["wind_speed_100m_lag1"].to_list() == [None, 1.0, 2.0, 3.0, 4.0]
    assert out["wind_speed_100m_lead1"].to_list() == [2.0, 3.0, 4.0, 5.0, None]
    # centered rolling window 3: index 2 = mean(2,3,4)
    assert out["wind_speed_100m_roll3_mean"].to_list()[2] == 3.0
    assert out["wind_speed_100m_roll3_max"].to_list()[2] == 4.0


def test_nwp_trajectory_noop_without_driver():
    df = pl.DataFrame({"temperature_2m": [1.0, 2.0]})
    out = blocks.nwp_trajectory(df)
    assert out.columns == ["temperature_2m"]


def test_nwp_trajectory_registered_for_both_kinds():
    from openenergy.features.registry import available_blocks
    assert "nwp_trajectory" in available_blocks("wind")
    assert "nwp_trajectory" in available_blocks("solar")


def test_wind_physics_ext_power_curve_and_cutout():
    df = pl.DataFrame({
        "wind_speed_100m": [2.0, 12.0, 30.0],  # below cut-in, rated, above cut-out
        "wind_direction_100m": [0.0, 90.0, 180.0],
    })
    out = blocks.wind_physics_ext(df)
    pc = out["wind_pc"].to_list()
    assert pc[0] == 0.0
    assert abs(pc[1] - 1.0) < 1e-9
    assert pc[2] == 0.0
    assert out["wind_cutout"].to_list() == [0.0, 0.0, 1.0]
    u = np.array(out["wind_u"].to_list())
    v = np.array(out["wind_v"].to_list())
    assert np.allclose(np.sqrt(u**2 + v**2), [2.0, 12.0, 30.0])


def test_wind_physics_ext_noop_without_speed():
    df = pl.DataFrame({"temperature_2m": [1.0]})
    assert blocks.wind_physics_ext(df).columns == ["temperature_2m"]


def test_wind_physics_ext_registered_for_wind_only():
    from openenergy.features.registry import available_blocks
    assert "wind_physics_ext" in available_blocks("wind")
    assert "wind_physics_ext" not in available_blocks("solar")
