from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order

_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42,
}


class QuantileGBMModel(ModelWrapper):
    family = "quantile_gbm"

    def __init__(
        self,
        params: dict | None = None,
        *,
        quantiles: tuple[float, float] = (0.1, 0.9),
    ) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.quantiles = quantiles
        self._p50: GradientBoostingRegressor | None = None
        self._lo: GradientBoostingRegressor | None = None
        self._hi: GradientBoostingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        if self._fit_constant_target(y):
            return
        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._p50 = GradientBoostingRegressor(loss="squared_error", **self.params).fit(
            X, y, sample_weight=sample_weight
        )
        qlo, qhi = self.quantiles
        self._lo = GradientBoostingRegressor(loss="quantile", alpha=qlo, **self.params).fit(
            X, y, sample_weight=sample_weight
        )
        self._hi = GradientBoostingRegressor(loss="quantile", alpha=qhi, **self.params).fit(
            X, y, sample_weight=sample_weight
        )

    def predict(self, X: np.ndarray) -> Prediction:
        constant = self._constant_prediction(X, with_bands=True)
        if constant is not None:
            return constant
        assert self._p50 is not None and self._lo is not None and self._hi is not None
        X = np.nan_to_num(np.asarray(X, dtype=float))
        return enforce_quantile_order(
            Prediction(
                p50=self._p50.predict(X),
                p10=self._lo.predict(X),
                p90=self._hi.predict(X),
            )
        )
