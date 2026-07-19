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
