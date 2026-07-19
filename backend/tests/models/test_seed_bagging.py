import numpy as np

from openenergy.models.lightgbm_model import LightGBMModel
from openenergy.models.seed_bagging import SeedBaggingModel


def _noisy(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.clip(
        0.5 + 0.3 * np.sin(2 * X[:, 0]) + rng.normal(scale=0.4, size=n),
        0, 1,
    )
    return X, y


_PARAMS = {"n_estimators": 60, "num_leaves": 15, "learning_rate": 0.1, "verbose": -1}


def test_seed_bagging_is_deterministic():
    X, y = _noisy(160, seed=0)
    k = 120
    a = SeedBaggingModel(params=_PARAMS, base_seed=7, n_seeds=4)
    b = SeedBaggingModel(params=_PARAMS, base_seed=7, n_seeds=4)
    a.fit(X[:k], y[:k])
    b.fit(X[:k], y[:k])
    pa, pb = a.predict(X[k:]), b.predict(X[k:])
    assert np.array_equal(pa.p50, pb.p50)
    assert np.array_equal(pa.p10, pb.p10)
    assert np.array_equal(pa.p90, pb.p90)


def test_seed_bagging_bands_ordered():
    X, y = _noisy(160, seed=1)
    k = 120
    m = SeedBaggingModel(params=_PARAMS, base_seed=3, n_seeds=5)
    m.fit(X[:k], y[:k])
    pred = m.predict(X[k:])
    assert pred.p10 is not None and pred.p90 is not None
    assert np.all(pred.p10 <= pred.p50)
    assert np.all(pred.p50 <= pred.p90)


def test_bagged_p50_variance_below_single_seed():
    """Averaging seeds shrinks the seed-induced variance of the p50 forecast."""
    X, y = _noisy(300, seed=2)
    k = 220
    Xtr, ytr, Xte = X[:k], y[:k], X[k:]

    n_bags = 12
    n_seeds = 5
    # Disjoint seed groups so each bag is an independent 5-seed average.
    single_preds = []
    bagged_preds = []
    for g in range(n_bags):
        seeds = range(g * n_seeds, g * n_seeds + n_seeds)
        member_p50s = []
        for s in seeds:
            sm = LightGBMModel(params={**_PARAMS, "random_state": s})
            sm.fit(Xtr, ytr)
            member_p50s.append(sm.predict(Xte).p50)
        member_p50s = np.array(member_p50s)
        single_preds.extend(member_p50s)  # each seed's own prediction
        bagged_preds.append(member_p50s.mean(axis=0))  # the bag average

    single_var = np.var(np.array(single_preds), axis=0).mean()
    bagged_var = np.var(np.array(bagged_preds), axis=0).mean()
    assert bagged_var < single_var
