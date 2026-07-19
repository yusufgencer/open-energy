import numpy as np
import polars as pl

from openenergy.features import blocks
from openenergy.features.registry import available_blocks
from openenergy.physics.solar import pv_model_chain_power


def _clear_day_gti(n: int = 24) -> np.ndarray:
    """Synthetic clear-day GTI: a daytime half-sine bell, zero at night."""
    hours = np.arange(n)
    x = (hours - 6.0) / 12.0  # daylight window ~06:00-18:00
    bell = np.sin(np.pi * np.clip(x, 0.0, 1.0))
    return 950.0 * bell


def test_pv_model_chain_power_zero_without_irradiance():
    gti = np.array([0.0, 0.0, 0.0])
    p = pv_model_chain_power(gti)
    assert np.allclose(p, 0.0)


def test_pv_model_chain_power_monotonic_in_irradiance():
    gti = np.array([0.0, 200.0, 500.0, 900.0])
    p = pv_model_chain_power(gti)
    assert np.all(np.diff(p) > 0.0)
    assert p[0] == 0.0


def test_p_physical_correlates_with_clear_day_production():
    gti = _clear_day_gti()
    temp_air = np.full_like(gti, 20.0)
    wind_speed = np.full_like(gti, 2.0)
    p_phys = pv_model_chain_power(gti, temp_air, wind_speed)

    # Synthetic clear-day production: PVWatts-shaped, mild temperature derate,
    # small multiplicative noise. Physics-as-feature should track it closely.
    rng = np.random.default_rng(7)
    cell = temp_air + gti * 0.03
    production = (gti / 1000.0) * (1.0 - 0.004 * (cell - 25.0))
    production = np.clip(production, 0.0, None) * (1.0 + 0.02 * rng.standard_normal(gti.size))

    corr = np.corrcoef(p_phys, production)[0, 1]
    assert corr >= 0.9


def test_pv_model_chain_power_handles_missing_aux_and_nan():
    gti = np.array([0.0, 300.0, np.nan, 800.0])
    p = pv_model_chain_power(gti)  # no temp_air / wind_speed supplied
    assert np.all(np.isfinite(p))
    assert p[0] == 0.0
    assert p[2] == 0.0  # nan irradiance -> no power


def test_block_registered_for_solar_only():
    assert "pv_model_chain" in available_blocks("solar")
    assert "pv_model_chain" not in available_blocks("wind")


def test_block_adds_p_physical_when_gti_present():
    gti = _clear_day_gti()
    df = pl.DataFrame({
        "valid_time": list(range(gti.size)),
        "power_mw": [0.0] * gti.size,
        "global_tilted_irradiance": gti.tolist(),
        "temperature_2m": [20.0] * gti.size,
        "wind_speed_10m": [2.0] * gti.size,
    })
    out = blocks.pv_model_chain(df)
    assert "p_physical" in out.columns
    assert out.height == gti.size
    # row order/count preserved; original columns intact
    assert out["power_mw"].to_list() == [0.0] * gti.size
    p = np.asarray(out["p_physical"].to_list())
    assert p[0] == 0.0
    assert p.max() > 0.0


def test_block_noop_without_gti():
    df = pl.DataFrame({
        "valid_time": [0, 1, 2],
        "power_mw": [1.0, 2.0, 3.0],
        "shortwave_radiation": [100.0, 200.0, 300.0],
    })
    assert blocks.pv_model_chain(df).columns == df.columns
