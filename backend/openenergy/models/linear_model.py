from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from openenergy.models.base import ModelWrapper, Prediction


class RidgeModel(ModelWrapper):
    family = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler: StandardScaler | None = None
        self._model: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        self._scaler = StandardScaler().fit(X)        # train-only fit
        Xs = self._scaler.transform(X)
        self._model = Ridge(alpha=self.alpha).fit(Xs, y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> Prediction:
        assert self._scaler is not None and self._model is not None
        Xs = self._scaler.transform(np.nan_to_num(np.asarray(X, dtype=float)))
        return Prediction(p50=self._model.predict(Xs))
