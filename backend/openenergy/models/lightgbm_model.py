from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor

from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order

_DEFAULTS = {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "verbose": -1, "random_state": 42}


class LightGBMModel(ModelWrapper):
    family = "lightgbm"

    def __init__(self, params: dict | None = None, *, quantiles: tuple[float, float] = (0.1, 0.9)) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.quantiles = quantiles
        self._p50 = None
        self._lo = None
        self._hi = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._p50 = LGBMRegressor(objective="regression", **self.params).fit(
            X, y, sample_weight=sample_weight
        )
        qlo, qhi = self.quantiles
        self._lo = LGBMRegressor(objective="quantile", alpha=qlo, **self.params).fit(
            X, y, sample_weight=sample_weight
        )
        self._hi = LGBMRegressor(objective="quantile", alpha=qhi, **self.params).fit(
            X, y, sample_weight=sample_weight
        )

    def predict(self, X: np.ndarray) -> Prediction:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        return enforce_quantile_order(
            Prediction(p50=self._p50.predict(X), p10=self._lo.predict(X), p90=self._hi.predict(X))
        )
