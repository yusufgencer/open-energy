from openenergy.features.registry import REGISTRY, available_blocks, get_block


def test_wind_blocks_available():
    wb = available_blocks("wind")
    assert "wind_power" in wb and "cyclical_time" in wb
    assert "clear_sky_index" not in wb


def test_solar_blocks_available():
    sb = available_blocks("solar")
    assert "clear_sky_index" in sb and "cyclical_time" in sb
    assert "wind_power" not in sb


def test_get_block_returns_callable():
    b = get_block("wind_power")
    assert callable(b.func)


def test_get_block_unknown_raises_value_error():
    import pytest
    with pytest.raises(ValueError):
        get_block("nope")
