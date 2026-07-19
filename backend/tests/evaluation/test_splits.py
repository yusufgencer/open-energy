import numpy as np
import pytest
from openenergy.evaluation import splits


def test_walk_forward_holds_out_tail():
    tr, te = splits.walk_forward_split(10, test_size=3)
    assert list(te) == [7, 8, 9]
    assert list(tr) == [0, 1, 2, 3, 4, 5, 6]


def test_walk_forward_too_large_raises():
    with pytest.raises(ValueError):
        splits.walk_forward_split(5, test_size=5)


def test_blocked_cv_embargo_gap_and_disjoint():
    folds = splits.blocked_cv_splits(100, n_splits=4, embargo=5)
    assert len(folds) == 4
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))            # ayrık
        assert tr.max() < te.min() - 5                 # embargo boşluğu
        assert (np.diff(te) == 1).all()                # ardışık test bloğu


def test_blocked_cv_tests_are_ordered_nonoverlapping():
    folds = splits.blocked_cv_splits(100, n_splits=4, embargo=2)
    test_mins = [te.min() for _, te in folds]
    assert test_mins == sorted(test_mins)
    seen = set()
    for _, te in folds:
        assert seen.isdisjoint(set(te)); seen |= set(te)


def test_blocked_oof_splits_meet_coverage_and_embargo():
    folds = splits.blocked_oof_splits(
        173, n_splits=3, embargo=24, min_coverage_count=116
    )
    covered = np.concatenate([te for _, te in folds])
    assert np.unique(covered).size >= 116
    for tr, te in folds:
        assert set(tr).isdisjoint(te)
        assert tr.max() + 24 < te.min()


def test_blocked_oof_splits_reject_impossible_coverage():
    with pytest.raises(ValueError, match="coverage"):
        splits.blocked_oof_splits(
            60, n_splits=3, embargo=24, min_coverage_count=40
        )


# --- F1-t2: group-by-valid_time split (no cross-lead leakage when rows are pooled) ---

def test_group_time_split_no_valid_time_straddles_boundary():
    # Two rows share each valid_time (e.g. lead 24 and lead 48 of the same time).
    vt = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    tr, te = splits.group_time_split(vt, test_fraction=0.4)
    tr_times = set(vt[tr].tolist())
    te_times = set(vt[te].tolist())
    # No valid_time appears in BOTH partitions → a pooled row's twin at another
    # lead can never leak across the boundary.
    assert tr_times.isdisjoint(te_times)
    # Every row is assigned exactly once, partition is complete.
    assert sorted(tr.tolist() + te.tolist()) == list(range(10))
    # Test set is the LATEST unique valid_times (temporal hold-out).
    assert max(tr_times) < min(te_times)


def test_group_time_split_datetime_and_fraction():
    import datetime as dt
    base = dt.datetime(2024, 1, 1)
    times = [base + dt.timedelta(hours=h) for h in range(5) for _ in range(3)]
    vt = np.array(times, dtype="datetime64[us]")
    tr, te = splits.group_time_split(vt, test_fraction=0.2)
    # 5 unique times, 20% → last 1 time (3 rows) held out
    assert len(te) == 3
    assert len(tr) == 12
    assert set(vt[tr].tolist()).isdisjoint(set(vt[te].tolist()))


def test_group_time_split_rejects_degenerate_fraction():
    vt = np.array([0, 0, 1, 1])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            splits.group_time_split(vt, test_fraction=bad)


def test_rolling_origin_no_valid_time_straddles_and_window_fixed():
    # Duplicate each timestamp to model pooled lead rows.  The split contract is
    # expressed in unique valid-time units, so duplicates must travel together.
    valid_time = np.repeat(np.arange(80), 2)
    folds = splits.rolling_origin_splits(
        valid_time, n_splits=3, train_size=32, test_size=8, embargo=4
    )

    assert len(folds) == 3
    train_window_lengths = []
    seen_test_times: set[int] = set()
    for train_idx, test_idx in folds:
        train_times = set(valid_time[train_idx].tolist())
        test_times = set(valid_time[test_idx].tolist())
        assert train_times.isdisjoint(test_times)
        assert seen_test_times.isdisjoint(test_times)
        assert max(train_times) + 4 < min(test_times)
        assert all(np.sum(valid_time[train_idx] == t) == 2 for t in train_times)
        assert all(np.sum(valid_time[test_idx] == t) == 2 for t in test_times)
        train_window_lengths.append(len(train_times))
        seen_test_times.update(test_times)

    assert train_window_lengths == [32, 32, 32]
