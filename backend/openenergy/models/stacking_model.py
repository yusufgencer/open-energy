from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openenergy.models.base import ModelWrapper, Prediction

_DEFAULTS = {
    "rf_n_estimators": 100,
    "gb_n_estimators": 100,
    "gb_learning_rate": 0.05,
    "final_alpha": 1.0,
    "n_jobs": 1,
}


class StackingModel(ModelWrapper):
    family = "stacking"

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self._model: StackingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        estimators = [
            (
                "ridge",
                make_pipeline(StandardScaler(), Ridge(alpha=self.params["final_alpha"])),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=self.params["rf_n_estimators"],
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=self.params["n_jobs"],
                ),
            ),
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=self.params["gb_n_estimators"],
                    learning_rate=self.params["gb_learning_rate"],
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
        self._model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=self.params["final_alpha"]),
            cv=3,
            n_jobs=self.params["n_jobs"],
        # StackingRegressor forwards sample_weight to every base estimator; the Ridge
        # pipeline needs nested fit params for that, so V1 keeps stacking unweighted.
        ).fit(X, y)

    def predict(self, X: np.ndarray) -> Prediction:
        assert self._model is not None
        X = np.nan_to_num(np.asarray(X, dtype=float))
        return Prediction(p50=self._model.predict(X))
