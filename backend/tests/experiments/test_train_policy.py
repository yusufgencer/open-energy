import numpy as np
import optuna

from openenergy.experiments.search_space import suggest
from openenergy.experiments.train_policy import (
    POLICY_NAMES,
    SOLAR_POLICY_NAMES,
    sample_weights,
    select_train_indices,
)


def _times(days=200):
    return np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(days)]
    )


def test_expanding_all_keeps_all_train_indices():
    vt = _times()
    idx = np.arange(100)
    out = select_train_indices(vt, idx, "expanding_all")
    assert np.array_equal(out, idx)


def test_rolling_90d_keeps_recent_window_only():
    vt = _times()
    idx = np.arange(160)
    out = select_train_indices(vt, idx, "rolling_90d")
    assert out.min() >= 69
    assert out.max() == 159


def test_recent_weighted_returns_increasing_weights():
    vt = _times()
    idx = np.arange(10)
    weights = sample_weights(vt, idx, "recent_weighted")
    assert weights is not None
    assert weights[0] < weights[-1]
    assert np.isclose(weights[0], 0.5)
    assert np.isclose(weights[-1], 1.5)


def test_pcs_weighted_weights_proportional_to_pcs():
    vt = _times()
    idx = np.arange(10)
    pcs = np.arange(10, dtype=float)  # 0..9 clear-sky power
    weights = sample_weights(vt, idx, "pcs_weighted", pcs=pcs)
    assert weights is not None
    # positive clear-sky rows are proportional to P_cs
    hi = pcs[idx] > 0
    ratios = weights[hi] / pcs[idx][hi]
    assert np.allclose(ratios, ratios[0])
    # peak (high irradiance) rows carry more weight than dawn rows
    assert weights[-1] > weights[1]


def test_pcs_weighted_floor_is_positive_at_night():
    vt = _times()
    idx = np.arange(5)
    pcs = np.array([0.0, 0.0, 5.0, 10.0, 0.0])  # night rows are 0
    weights = sample_weights(vt, idx, "pcs_weighted", pcs=pcs)
    assert weights is not None
    assert np.all(weights > 0.0)  # floor > 0, no zero-weight rows
    assert weights[3] == weights.max()


def test_pcs_weighted_without_pcs_returns_none():
    vt = _times()
    idx = np.arange(5)
    assert sample_weights(vt, idx, "pcs_weighted") is None


def test_pcs_weighted_selects_all_train_indices():
    vt = _times()
    idx = np.arange(100)
    out = select_train_indices(vt, idx, "pcs_weighted")
    assert np.array_equal(out, idx)


def test_non_pcs_policy_ignores_pcs():
    vt = _times()
    idx = np.arange(10)
    pcs = np.arange(10, dtype=float)
    assert sample_weights(vt, idx, "expanding_all", pcs=pcs) is None


def _solar_fixed_trial(train_policy):
    from openenergy.features.registry import available_blocks

    params = {
        "model_family": "ridge",
        "target_policy": "capacity_norm",
        "train_policy": train_policy,
        "feature_family": "solar_clear_sky",
        "alpha": 1.0,
    }
    for blk in available_blocks("solar"):
        params[f"blk_{blk}"] = (blk == "cyclical_time")
    return optuna.trial.FixedTrial(params)


def test_solar_search_can_select_pcs_weighting():
    assert "pcs_weighted" in SOLAR_POLICY_NAMES
    cfg = suggest(_solar_fixed_trial("pcs_weighted"), "solar", ["ecmwf"])
    assert cfg.train_policy == "pcs_weighted"


def test_wind_train_policy_unaffected_by_pcs_weighting():
    # pcs_weighted is a solar-only axis; wind's train_policy choices exclude it.
    assert "pcs_weighted" not in POLICY_NAMES
