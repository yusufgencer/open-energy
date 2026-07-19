from __future__ import annotations

import numpy as np

from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order

_DEFAULTS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


class CatBoostModel(ModelWrapper):
    family = "catboost"

    def __init__(
        self,
        params: dict | None = None,
        *,
        quantiles: tuple[float, float] | None = None,
    ) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.quantiles = quantiles
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost paketi kurulu değil; `uv sync` çalıştırın.") from exc

        X = np.nan_to_num(np.asarray(X, dtype=float))
        params = dict(self.params)
        if self.quantiles is not None:
            qlo, qhi = self.quantiles
            # Fit all quantiles jointly with a single MultiQuantile head — one tree
            # structure shared across alphas, so the outputs are non-crossing by
            # construction (unlike independent per-quantile heads).
            params["loss_function"] = f"MultiQuantile:alpha={qlo},0.5,{qhi}"
        self._model = CatBoostRegressor(**params).fit(X, y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> Prediction:
        assert self._model is not None
        X = np.nan_to_num(np.asarray(X, dtype=float))
        raw = np.asarray(self._model.predict(X), dtype=float)
        if self.quantiles is None:
            return Prediction(p50=raw)
        # MultiQuantile output is (n_rows, 3) in the fitted alpha order (lo, 0.5, hi).
        return enforce_quantile_order(
            Prediction(p50=raw[:, 1], p10=raw[:, 0], p90=raw[:, 2])
        )
