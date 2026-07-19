from __future__ import annotations

import numpy as np

from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order
from openenergy.models.lightgbm_model import LightGBMModel


class SeedBaggingModel(ModelWrapper):
    """Average 3-5 seeds of a base LightGBM ("cheap alpha").

    Each member is the same base GBM fitted with a distinct ``random_state``
    (``base_seed + i``); the point forecast is the mean of the member p50s and
    the quantile bands are *Vincentized* — averaged per quantile level across the
    members (the empirical-quantile counterpart of averaging point forecasts).
    Ordering is enforced once after combination. Deterministic: identical
    ``params``/``base_seed``/``n_seeds`` reproduce identical predictions.
    """

    family = "seed_bagged_lightgbm"

    def __init__(
        self,
        params: dict | None = None,
        *,
        quantiles: tuple[float, float] = (0.1, 0.9),
        n_seeds: int = 5,
        base_seed: int = 42,
    ) -> None:
        self.params = dict(params or {})
        self.quantiles = quantiles
        self.n_seeds = int(np.clip(n_seeds, 3, 5))
        self.base_seed = base_seed
        self._members: list[LightGBMModel] = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        self._members = []
        for i in range(self.n_seeds):
            member = LightGBMModel(
                params={**self.params, "random_state": self.base_seed + i},
                quantiles=self.quantiles,
            )
            member.fit(X, y, sample_weight=sample_weight)
            self._members.append(member)

    def predict(self, X: np.ndarray) -> Prediction:
        preds = [m.predict(X) for m in self._members]
        p50 = np.mean([p.p50 for p in preds], axis=0)
        p10 = (
            np.mean([p.p10 for p in preds], axis=0)
            if all(p.p10 is not None for p in preds)
            else None
        )
        p90 = (
            np.mean([p.p90 for p in preds], axis=0)
            if all(p.p90 is not None for p in preds)
            else None
        )
        return enforce_quantile_order(Prediction(p50=p50, p10=p10, p90=p90))
