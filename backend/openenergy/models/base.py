from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Prediction:
    p50: np.ndarray
    p10: np.ndarray | None = None
    p90: np.ndarray | None = None


def enforce_quantile_order(pred: Prediction) -> Prediction:
    """Return a Prediction with non-crossing quantiles (p10 <= p50 <= p90).

    Uses the "clamp-to-p50" rule: p10 = min(p10, p50), p90 = max(p90, p50).
    The point forecast p50 is never modified. Missing bands pass through
    unchanged. Independent quantile heads (LightGBM, QuantileGBM, RF) can
    cross; this is the single canonical fix applied at every model, ensemble
    and serving layer so no crossing prediction is ever scored, stored or served.
    """
    if pred.p10 is None and pred.p90 is None:
        return pred
    p10 = np.minimum(pred.p10, pred.p50) if pred.p10 is not None else None
    p90 = np.maximum(pred.p90, pred.p50) if pred.p90 is not None else None
    return Prediction(p50=pred.p50, p10=p10, p90=p90)


class ModelWrapper(ABC):
    family: str = "base"

    def _fit_constant_target(self, y: np.ndarray) -> bool:
        """Record a degenerate training target and skip the underlying estimator.

        Constant folds occur naturally for night-time solar production and very
        short validation windows.  Treating them as a valid constant predictor
        keeps every model family deterministic and avoids estimator-specific
        failures (notably CatBoost's "All train targets are equal" error).
        """
        values = np.asarray(y, dtype=float).reshape(-1)
        self._constant_target = (
            float(values[0])
            if values.size > 0 and np.all(values == values[0])
            else None
        )
        return self._constant_target is not None

    def _constant_prediction(
        self,
        X: np.ndarray,
        *,
        with_bands: bool = False,
    ) -> Prediction | None:
        value = getattr(self, "_constant_target", None)
        if value is None:
            return None
        p50 = np.full(len(X), value, dtype=float)
        if with_bands:
            return Prediction(p50=p50, p10=p50.copy(), p90=p50.copy())
        return Prediction(p50=p50)

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction: ...

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "ModelWrapper":
        return pickle.loads(Path(path).read_bytes())
