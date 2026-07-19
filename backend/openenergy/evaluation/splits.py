from __future__ import annotations

import numpy as np


def walk_forward_split(n: int, *, test_size: int) -> tuple[np.ndarray, np.ndarray]:
    if test_size >= n or test_size <= 0:
        raise ValueError("0 < test_size < n olmalı")
    cut = n - test_size
    return np.arange(0, cut), np.arange(cut, n)


def group_time_split(valid_time: np.ndarray, *, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Temporal hold-out that splits by *unique valid_time*, not by row (F1-t2).

    When rows are pooled over leads (the same ``valid_time`` appears once per
    lead), a plain positional split would place lead-24 of time ``T`` in train and
    lead-48 of the same ``T`` in test — cross-lead leakage. This split assigns
    every row of a given ``valid_time`` to the same side: the latest
    ``test_fraction`` of unique valid_times (and all their rows) form the test set,
    the earlier ones form train. The two time-sets are therefore disjoint, so no
    valid_time (hence no pooled twin) ever straddles the boundary.

    Returns positional ``(train_idx, test_idx)`` into ``valid_time``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("0 < test_fraction < 1 olmalı")
    valid_time = np.asarray(valid_time)
    uniq = np.unique(valid_time)  # np.unique returns sorted ascending
    n_u = uniq.size
    if n_u < 2:
        raise ValueError("en az 2 farklı valid_time gerekli")
    n_test = max(1, int(round(n_u * test_fraction)))
    n_test = min(n_test, n_u - 1)  # keep at least one train time
    test_times = set(uniq[n_u - n_test:].tolist())
    is_test = np.array([v in test_times for v in valid_time.tolist()], dtype=bool)
    idx = np.arange(valid_time.shape[0])
    return idx[~is_test], idx[is_test]


def rolling_origin_splits(
    valid_time: np.ndarray,
    *,
    n_splits: int,
    test_size: int,
    embargo: int,
    train_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed-window rolling-origin outer folds grouped by ``valid_time``.

    ``test_size``, ``embargo`` and ``train_size`` count *unique valid times*,
    rather than rows.  This is important for pooled datasets where several rows
    can share one valid time: every such row stays on the same side of a fold.
    The folds are anchored at the end of the series, their test blocks are
    ordered and non-overlapping, and every training window has exactly the same
    number of unique times.

    When ``train_size`` is omitted, the largest window that fits all requested
    origins is used.  Positional indices into the original array are returned.
    """
    valid_time = np.asarray(valid_time)
    if valid_time.ndim != 1:
        raise ValueError("valid_time tek boyutlu olmalı")
    if n_splits < 2:
        raise ValueError("n_splits >= 2 olmalı")
    if test_size <= 0:
        raise ValueError("test_size > 0 olmalı")
    if embargo < 0:
        raise ValueError("embargo >= 0 olmalı")

    unique_times = np.unique(valid_time)
    first_test = unique_times.size - n_splits * test_size
    max_train_size = first_test - embargo
    if train_size is None:
        train_size = max_train_size
    if train_size <= 0 or train_size > max_train_size:
        raise ValueError(
            "rolling-origin için train/test/embargo pencerelerine yetecek valid_time yok"
        )

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_no in range(n_splits):
        test_start = first_test + fold_no * test_size
        test_end = test_start + test_size
        train_end = test_start - embargo
        train_start = train_end - train_size
        train_times = unique_times[train_start:train_end]
        test_times = unique_times[test_start:test_end]
        train_mask = np.isin(valid_time, train_times)
        test_mask = np.isin(valid_time, test_times)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("rolling-origin boş fold üretti")
        out.append((train_idx, test_idx))
    return out


def blocked_cv_splits(n: int, *, n_splits: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2 or n_splits >= n:
        raise ValueError("2 <= n_splits < n olmalı")
    fold = n // (n_splits + 1)
    if fold <= embargo:
        raise ValueError("fold boyutu embargo'dan büyük olmalı (n az / embargo çok)")
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        test_start = fold * k
        test_end = n if k == n_splits else fold * (k + 1)
        train_end = test_start - embargo
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        out.append((train_idx, test_idx))
    return out


def blocked_oof_splits(
    n: int,
    *,
    n_splits: int,
    embargo: int,
    min_coverage_count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds for a sufficiently large blocked OOF set.

    The requested split count is tried first. If its blocks are too short for
    the embargo, progressively fewer (and therefore wider) blocks are tried.
    Every returned validation block is strictly later than its training block
    and separated from it by ``embargo`` positions.

    ``min_coverage_count`` is absolute rather than a fraction because callers
    may reserve a trailing policy-selection holdout outside these ``n`` rows
    while still requiring OOF coverage relative to the full training window.
    """
    if min_coverage_count <= 0 or min_coverage_count > n:
        raise ValueError("0 < min_coverage_count <= n olmalı")
    upper = min(n_splits, n - 1)
    for candidate_splits in range(upper, 1, -1):
        try:
            folds = blocked_cv_splits(
                n, n_splits=candidate_splits, embargo=embargo
            )
        except ValueError:
            continue
        covered = np.unique(
            np.concatenate([test_idx for _, test_idx in folds])
        ).size
        if covered >= min_coverage_count:
            return folds
    raise ValueError(
        "embargo güvenli blocked OOF fold'ları istenen coverage'ı karşılamıyor"
    )
