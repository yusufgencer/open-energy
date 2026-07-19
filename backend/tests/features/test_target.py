import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st
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


@settings(max_examples=200, deadline=None)
@given(
    capacity_mw=st.floats(min_value=0.1, max_value=1_000.0, allow_nan=False),
    clear_sky_fraction=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False),
    production_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_kpv_roundtrip_daylight_recovers_power(
    capacity_mw: float,
    clear_sky_fraction: float,
    production_fraction: float,
):
    """Every physically valid daylight value survives the kPV round-trip."""
    p_cs = capacity_mw * clear_sky_fraction
    power = p_cs * production_fraction

    recovered = kpv_inverse(kpv_forward([power], [p_cs]), [p_cs], capacity_mw)

    assert recovered[0] == pytest.approx(power, rel=1e-10, abs=1e-10)


@given(
    capacity_mw=st.floats(min_value=0.1, max_value=1_000.0, allow_nan=False),
    y_prime=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        min_size=1,
        max_size=30,
    ),
    p_cs=st.lists(
        st.floats(min_value=0.0, max_value=2_000.0, allow_nan=False),
        min_size=1,
        max_size=30,
    ),
)
def test_kpv_inverse_stays_inside_physical_band(capacity_mw, y_prime, p_cs):
    size = min(len(y_prime), len(p_cs))
    served = kpv_inverse(y_prime[:size], p_cs[:size], capacity_mw)

    assert np.isfinite(served).all()
    assert np.all(served >= 0.0)
    assert np.all(served <= capacity_mw)


def test_kpv_forward_clips_ratio_and_night_never_divides():
    # ratio above clip_max is capped; night rows never divide by zero
    p_cs = np.array([0.0, 1.0, 2.0])
    power = np.array([5.0, 5.0, 5.0])  # 5/1=5 → clipped, 0-row must not blow up
    yprime = kpv_forward(power, p_cs, clip_max=1.5)
    assert np.isfinite(yprime).all()
    assert yprime[0] == 0.0
    assert yprime.max() <= 1.5 + 1e-12


def test_kpv_forward_integer_inputs_keep_float_precision_and_floor_is_night():
    yprime = kpv_forward(
        power_mw=[1, 1, -1],
        p_cs=[2, 1e-6, 2],
        floor=1e-6,
    )

    assert yprime.dtype == np.dtype("float64")
    np.testing.assert_array_equal(yprime, [0.5, 0.0, 0.0])


def test_kpv_forward_coerces_numeric_array_like_inputs():
    yprime = kpv_forward(power_mw=["1"], p_cs=["2"])

    np.testing.assert_array_equal(yprime, [0.5])


def test_kpv_inverse_never_exceeds_capacity():
    yprime = np.array([1.0, 1.2, 0.5])
    p_cs = np.array([20.0, 5.0, 4.0])  # p_cs*yprime could exceed capacity
    served = kpv_inverse(yprime, p_cs, capacity_mw=10.0)
    assert served.max() <= 10.0
    assert served.min() >= 0.0


def test_kpv_inverse_integer_inputs_return_float64():
    served = kpv_inverse([1, -1], [2, 2], capacity_mw=10.0)

    assert served.dtype == np.dtype("float64")
    np.testing.assert_array_equal(served, [2.0, 0.0])


def test_kpv_inverse_coerces_numeric_array_like_inputs():
    served = kpv_inverse(["1", "-1"], ["2", "2"], capacity_mw=10.0)

    np.testing.assert_array_equal(served, [2.0, 0.0])


def test_plant_geometry_as_dict_round_trips():
    g = PlantGeometry(lat=39.9, lon=32.8, tilt=30.0, azimuth=180.0)
    d = g.as_dict()
    assert d == {"lat": 39.9, "lon": 32.8, "tilt": 30.0, "azimuth": 180.0}
    assert PlantGeometry(**d) == g
