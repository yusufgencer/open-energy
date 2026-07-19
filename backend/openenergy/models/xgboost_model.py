from __future__ import annotations

import numpy as np

from openenergy.models.base import ModelWrapper, Prediction

_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostModel(ModelWrapper):
    family = "xgboost"

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost paketi kurulu değil; `uv sync` çalıştırın.") from exc

        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._model = XGBRegressor(**self.params).fit(X, y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> Prediction:
        assert self._model is not None
        X = np.nan_to_num(np.asarray(X, dtype=float))
        return Prediction(p50=self._model.predict(X))
