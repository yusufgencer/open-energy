import numpy as np
import polars as pl
import pytest
from openenergy.features.target import (
    PlantGeometry,
    kpv_forward,
    kpv_inverse,
    normalize_power,
)


def test_normalize_divides_by_capacity():
    df = pl.DataFrame({"power_mw": [0.0, 5.0, 10.0]})
    out = normalize_power(df, 10.0)
    assert out["power_norm"].to_list() == [0.0, 0.5, 1.0]


def test_zero_capacity_raises():
    with pytest.raises(ValueError):
        normalize_power(pl.DataFrame({"power_mw": [1.0]}), 0.0)


# --- kPV target transform (C1-t3) ---

def test_kpv_forward_is_daylight_ratio():
    p_cs = np.array([0.0, 2.0, 5.0, 8.0, 5.0, 2.0, 0.0])
    power = 0.7 * p_cs
    yprime = kpv_forward(power, p_cs)
    # daylight rows → constant clear-sky ratio; night rows (p_cs=0) → 0
    assert yprime[0] == 0.0 and yprime[-1] == 0.0
    np.testing.assert_allclose(yprime[1:-1], 0.7, rtol=1e-9)


def test_kpv_round_trip_identity():
    p_cs = np.array([0.0, 1.5, 4.0, 7.0, 4.0, 1.5, 0.0])
    power = 0.62 * p_cs  # production below the clear-sky envelope
    yprime = kpv_forward(power, p_cs)
    back = kpv_inverse(yprime, p_cs, capacity_mw=10.0)
    np.testing.assert_allclose(back, power, rtol=1e-9, atol=1e-9)


def test_kpv_forward_clips_ratio_and_night_never_divides():
    # ratio above clip_max is capped; night rows never divide by zero
    p_cs = np.array([0.0, 1.0, 2.0])
    power = np.array([5.0, 5.0, 5.0])  # 5/1=5 → clipped, 0-row must not blow up
    yprime = kpv_forward(power, p_cs, clip_max=1.5)
    assert np.isfinite(yprime).all()
    assert yprime[0] == 0.0
    assert yprime.max() <= 1.5 + 1e-12


def test_kpv_inverse_never_exceeds_capacity():
    yprime = np.array([1.0, 1.2, 0.5])
    p_cs = np.array([20.0, 5.0, 4.0])  # p_cs*yprime could exceed capacity
    served = kpv_inverse(yprime, p_cs, capacity_mw=10.0)
    assert served.max() <= 10.0
    assert served.min() >= 0.0


def test_plant_geometry_as_dict_round_trips():
    g = PlantGeometry(lat=39.9, lon=32.8, tilt=30.0, azimuth=180.0)
    d = g.as_dict()
    assert d == {"lat": 39.9, "lon": 32.8, "tilt": 30.0, "azimuth": 180.0}
    assert PlantGeometry(**d) == g
