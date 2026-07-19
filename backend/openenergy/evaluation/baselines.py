from __future__ import annotations

from typing import Callable

import numpy as np


def persistence_forecast(y_train: np.ndarray, n_test: int) -> np.ndarray:
    if len(y_train) == 0:
        raise ValueError("boş train")
    return np.full(n_test, y_train[-1], dtype=float)


def climatology_forecast(y_train: np.ndarray, n_test: int) -> np.ndarray:
    if len(y_train) == 0:
        raise ValueError("boş train")
    return np.full(n_test, float(np.mean(y_train)), dtype=float)


def diurnal_persistence_forecast(
    y_hist: np.ndarray, vt_hist: np.ndarray, vt_target: np.ndarray
) -> np.ndarray:
    """Same-hour-previous-day ("smart"-ish) persistence.

    For each target valid_time t, predict the observed value at t-24h looked up
    from the full known history (train + earlier test actuals). Falls back to the
    historical mean when no t-24h observation exists (e.g. first day). This is the
    honest solar/wind reference — flat persistence is a strawman at 24/48 h.
    """
    if len(y_hist) == 0:
        raise ValueError("boş train")
    lookup = {int(t): float(v) for t, v in zip(vt_hist.astype("datetime64[us]").astype("int64"), y_hist)}
    day = np.timedelta64(24, "h")
    prev_keys = (vt_target.astype("datetime64[us]") - day).astype("int64")
    fallback = float(np.mean(y_hist))
    return np.array([lookup.get(int(k), fallback) for k in prev_keys], dtype=float)


def baseline_errors(y_train: np.ndarray, y_test: np.ndarray,
                    metric_fn: Callable[[np.ndarray, np.ndarray], float],
                    *, vt_train: np.ndarray | None = None,
                    vt_test: np.ndarray | None = None,
                    train_mask: np.ndarray | None = None,
                    test_mask: np.ndarray | None = None) -> dict[str, float]:
    """Score references, optionally on the exact same post-split clean mask.

    Masks are deliberately accepted only after callers have formed their train
    and test slices.  This keeps the split universe invariant under QC flags.
    """
    train_keep = (
        np.ones(len(y_train), dtype=bool)
        if train_mask is None else np.asarray(train_mask, dtype=bool)
    )
    test_keep = (
        np.ones(len(y_test), dtype=bool)
        if test_mask is None else np.asarray(test_mask, dtype=bool)
    )
    y_train_clean = np.asarray(y_train)[train_keep]
    if len(y_train_clean) == 0 or not np.any(test_keep):
        return {}
    n = len(y_test)
    errs = {
        "persistence": metric_fn(
            np.asarray(y_test)[test_keep],
            persistence_forecast(y_train_clean, n)[test_keep],
        ),
        "climatology": metric_fn(
            np.asarray(y_test)[test_keep],
            climatology_forecast(y_train_clean, n)[test_keep],
        ),
    }
    if vt_train is not None and vt_test is not None:
        y_hist = np.concatenate([np.asarray(y_train)[train_keep], np.asarray(y_test)[test_keep]])
        vt_hist = np.concatenate([np.asarray(vt_train)[train_keep], np.asarray(vt_test)[test_keep]])
        errs["diurnal_persistence"] = metric_fn(
            np.asarray(y_test)[test_keep],
            diurnal_persistence_forecast(
                y_hist, vt_hist, np.asarray(vt_test)[test_keep]
            ),
        )
    return errs
