from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from openenergy.models.base import ModelWrapper, Prediction, enforce_quantile_order

_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


class RandomForestModel(ModelWrapper):
    """Random forest with true Quantile Regression Forest (QRF) bands.

    Point forecast (p50) is the standard forest mean. The quantile bands are
    extracted from the *pooled leaf empirical distribution* (Meinshausen 2006):
    for a query point, each tree contributes the training targets that share its
    leaf, weighted so every tree counts equally. Empirical quantiles of that
    pooled distribution recover the full conditional spread — unlike the variance
    of per-tree means, which collapses as ``n_estimators`` grows and badly
    understates the band width. Quantiles of a shared distribution are
    non-crossing by construction; ``enforce_quantile_order`` is applied as the
    canonical safety clamp.
    """

    family = "random_forest"

    def __init__(
        self,
        params: dict | None = None,
        *,
        quantiles: tuple[float, float] = (0.1, 0.9),
    ) -> None:
        self.params = {**_DEFAULTS, **(params or {})}
        self.quantiles = quantiles
        self._model: RandomForestRegressor | None = None
        self._y_train: np.ndarray | None = None
        # Per tree: mapping leaf_id -> (y-values in that leaf, count).
        self._leaf_map: list[dict[int, tuple[np.ndarray, int]]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        if self._fit_constant_target(y):
            return
        X = np.nan_to_num(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self._model = RandomForestRegressor(**self.params).fit(X, y, sample_weight=sample_weight)
        self._y_train = y
        # leaves: (n_train, n_trees) leaf index of each training row in each tree.
        leaves = self._model.apply(X)
        self._leaf_map = []
        for t in range(leaves.shape[1]):
            col = leaves[:, t]
            order = np.argsort(col, kind="stable")
            sorted_leaves = col[order]
            uniq, starts = np.unique(sorted_leaves, return_index=True)
            tree_map: dict[int, tuple[np.ndarray, int]] = {}
            for idx, leaf_id in enumerate(uniq):
                end = starts[idx + 1] if idx + 1 < len(starts) else len(order)
                members = order[starts[idx]:end]
                tree_map[int(leaf_id)] = (y[members], len(members))
            self._leaf_map.append(tree_map)

    def predict(self, X: np.ndarray) -> Prediction:
        constant = self._constant_prediction(X, with_bands=True)
        if constant is not None:
            return constant
        assert self._model is not None and self._y_train is not None
        X = np.nan_to_num(np.asarray(X, dtype=float))
        p50 = self._model.predict(X)

        qlo, qhi = self.quantiles
        test_leaves = self._model.apply(X)  # (n_test, n_trees)
        n_test, n_trees = test_leaves.shape

        p10 = np.empty(n_test)
        p90 = np.empty(n_test)
        for i in range(n_test):
            # Pooled leaf empirical distribution: concatenate the y-values sharing
            # this query's leaf in each tree; each member is weighted 1/leaf_size so
            # every tree contributes equally (Meinshausen QRF weights).
            vals_list: list[np.ndarray] = []
            wts_list: list[np.ndarray] = []
            for t in range(n_trees):
                members, count = self._leaf_map[t].get(int(test_leaves[i, t]), (None, 0))
                if count == 0:
                    continue
                vals_list.append(members)
                wts_list.append(np.full(count, 1.0 / count))
            if not vals_list:
                p10[i] = p90[i] = p50[i]
                continue
            vals = np.concatenate(vals_list)
            wts = np.concatenate(wts_list)
            p10[i] = _weighted_quantile(vals, wts, qlo)
            p90[i] = _weighted_quantile(vals, wts, qhi)

        return enforce_quantile_order(Prediction(p50=p50, p10=p10, p90=p90))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted empirical quantile via cumulative-weight interpolation."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(v[-1])
    # Midpoint cumulative positions in [0, 1].
    cum = (cw - 0.5 * w) / total
    return float(np.interp(q, cum, v))
