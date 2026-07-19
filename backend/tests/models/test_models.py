import numpy as np
import pytest
from openenergy.models.base import Prediction, enforce_quantile_order
from openenergy.models.lightgbm_model import LightGBMModel
from openenergy.models.linear_model import RidgeModel
from openenergy.models.quantile_gbm_model import QuantileGBMModel
from openenergy.models.random_forest_model import RandomForestModel
from openenergy.models.stacking_model import StackingModel
from openenergy.models.xgboost_model import XGBoostModel
from openenergy.models.catboost_model import CatBoostModel


def test_enforce_quantile_order_clamps_crossings_to_p50():
    pred = Prediction(
        p50=np.array([3.0, 5.0]),
        p10=np.array([4.0, 2.0]),  # first row: p10 crosses above p50
        p90=np.array([1.0, 8.0]),  # first row: p90 crosses below p50
    )
    out = enforce_quantile_order(pred)
    assert np.array_equal(out.p50, [3.0, 5.0])  # p50 never modified
    assert np.array_equal(out.p10, [3.0, 2.0])  # min(p10, p50)
    assert np.array_equal(out.p90, [3.0, 8.0])  # max(p90, p50)
    assert np.all(out.p10 <= out.p50) and np.all(out.p50 <= out.p90)


def test_enforce_quantile_order_passthrough_when_no_bands():
    pred = Prediction(p50=np.array([1.0, 2.0]))
    out = enforce_quantile_order(pred)
    assert out.p10 is None and out.p90 is None
    assert np.array_equal(out.p50, [1.0, 2.0])


def _hetero_data(n, seed):
    """Heteroscedastic high-noise target that induces raw quantile crossing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.clip(
        0.5 + 0.4 * np.sin(3 * X[:, 0]) + rng.normal(scale=0.4 + 0.3 * np.abs(X[:, 1]), size=n),
        0, 1,
    )
    return X, y


@pytest.mark.parametrize(
    ("name", "factory", "seed", "n"),
    [
        (
            "lgbm",
            lambda: LightGBMModel(
                params={"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1, "verbose": -1},
                quantiles=(0.1, 0.9),
            ),
            11, 140,
        ),
        (
            "qgbm",
            lambda: QuantileGBMModel(params={"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}),
            3, 90,
        ),
        (
            "rf",
            lambda: RandomForestModel(params={"n_estimators": 30, "max_depth": 10, "random_state": 1, "n_jobs": 1}),
            1, 90,
        ),
    ],
)
def test_quantile_models_never_cross(name, factory, seed, n):
    # Data/seed chosen so the raw quantile heads DO cross without the fix.
    X, y = _hetero_data(n, seed)
    k = int(n * 0.7)
    m = factory()
    m.fit(X[:k], y[:k])
    pred = m.predict(X[k:])
    assert np.all(pred.p10 <= pred.p50), f"{name}: p10 crosses above p50"
    assert np.all(pred.p50 <= pred.p90), f"{name}: p90 crosses below p50"


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.clip(0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1], 0, 1)
    return X, y


def test_ridge_learns_linear_signal():
    X, y = _data()
    m = RidgeModel(alpha=0.1)
    m.fit(X[:150], y[:150])
    pred = m.predict(X[150:])
    err = np.sqrt(np.mean((pred.p50 - y[150:]) ** 2))
    assert err < 0.1
    assert pred.p10 is None


def test_lightgbm_predicts_quantile_band():
    X, y = _data()
    m = LightGBMModel(params={"n_estimators": 50, "num_leaves": 7, "verbose": -1}, quantiles=(0.1, 0.9))
    m.fit(X[:150], y[:150])
    pred = m.predict(X[150:])
    assert pred.p10 is not None and pred.p90 is not None
    # p10 <= p90 çoğunlukla (monotonluk garanti değil ama ortalama band pozitif)
    assert np.mean(pred.p90 - pred.p10) > 0
    assert pred.p50.shape == (50,)


def test_random_forest_predicts_tree_quantile_band():
    X, y = _data(n=120)
    m = RandomForestModel(params={"n_estimators": 25, "max_depth": 6, "random_state": 42, "n_jobs": 1})
    m.fit(X[:90], y[:90])
    pred = m.predict(X[90:])
    assert pred.p10 is not None and pred.p90 is not None
    assert pred.p50.shape == (30,)
    assert np.mean(pred.p90 - pred.p10) >= 0


def test_random_forest_qrf_bands_widen_with_noise():
    """True QRF extracts spread from the pooled leaf empirical distribution:
    bands must be wider where the target noise is larger. The old per-tree-mean
    approach (variance of tree means, which shrinks with n_estimators) fails this.
    """
    rng = np.random.default_rng(7)
    n = 4000
    x_sig = rng.uniform(-2, 2, size=n)  # signal driver
    x_noise = rng.uniform(0.0, 1.0, size=n)  # heteroscedastic driver in [0,1]
    scale = 0.05 + 1.5 * x_noise  # noise widens with x_noise
    y = np.sin(x_sig) + rng.normal(scale=scale, size=n)
    X = np.column_stack([x_sig, x_noise])

    k = int(n * 0.75)
    m = RandomForestModel(
        params={"n_estimators": 200, "min_samples_leaf": 20, "random_state": 0, "n_jobs": 1},
    )
    m.fit(X[:k], y[:k])
    pred = m.predict(X[k:])
    width = pred.p90 - pred.p10
    y_test = y[k:]

    xn_test = x_noise[k:]
    low = xn_test < 0.33
    high = xn_test > 0.66
    assert low.sum() > 20 and high.sum() > 20
    # Bands in the high-noise region must be clearly wider than in the low-noise region.
    assert width[high].mean() > 1.5 * width[low].mean()
    # QRF extracts the full leaf spread, so the (0.1, 0.9) band achieves near-nominal
    # 80% coverage. The old per-tree-mean bands are far too narrow (~0.37 coverage here).
    coverage = np.mean((y_test >= pred.p10) & (y_test <= pred.p90))
    assert coverage > 0.70
    # Non-crossing by construction.
    assert np.all(pred.p10 <= pred.p50)
    assert np.all(pred.p50 <= pred.p90)


def test_quantile_gbm_predicts_quantile_band():
    X, y = _data(n=120)
    m = QuantileGBMModel(params={"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05})
    m.fit(X[:90], y[:90])
    pred = m.predict(X[90:])
    assert pred.p10 is not None and pred.p90 is not None
    assert pred.p50.shape == (30,)


def test_catboost_multiquantile_predicts_ordered_band():
    pytest.importorskip("catboost")
    X, y = _hetero_data(120, seed=7)
    m = CatBoostModel(
        params={"iterations": 60, "depth": 3, "learning_rate": 0.1, "verbose": False},
        quantiles=(0.1, 0.9),
    )
    m.fit(X[:90], y[:90])
    pred = m.predict(X[90:])
    assert pred.p10 is not None and pred.p90 is not None
    assert pred.p50.shape == (30,)
    # MultiQuantile is structurally non-crossing; verify ordered p10<=p50<=p90.
    assert np.all(pred.p10 <= pred.p50)
    assert np.all(pred.p50 <= pred.p90)
    assert np.mean(pred.p90 - pred.p10) > 0


def test_stacking_predicts_p50():
    X, y = _data(n=120)
    m = StackingModel(params={
        "rf_n_estimators": 10,
        "gb_n_estimators": 10,
        "gb_learning_rate": 0.05,
        "final_alpha": 1.0,
    })
    m.fit(X[:90], y[:90])
    pred = m.predict(X[90:])
    assert pred.p50.shape == (30,)
    assert pred.p10 is None


@pytest.mark.parametrize(
    ("model_cls", "params"),
    [
        (XGBoostModel, {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.05, "n_jobs": 1}),
        (CatBoostModel, {"iterations": 20, "depth": 3, "learning_rate": 0.05, "verbose": False}),
    ],
)
def test_optional_boosters_predict_p50(model_cls, params):
    pytest.importorskip("xgboost" if model_cls is XGBoostModel else "catboost")
    X, y = _data(n=120)
    m = model_cls(params=params)
    m.fit(X[:90], y[:90])
    pred = m.predict(X[90:])
    assert pred.p50.shape == (30,)
    assert pred.p10 is None
