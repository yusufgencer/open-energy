import datetime as dt

import polars as pl

from openenergy.features import spatial


def _pt(values):
    vt = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 1)]
    return pl.DataFrame({"valid_time": vt, "ws": values},
                        schema_overrides={"valid_time": pl.Datetime("us"), "ws": pl.Float64})


def test_aggregate_points_cross_point_stats():
    out = spatial.aggregate_points({0: _pt([1.0, 10.0]), 1: _pt([3.0, 20.0])}, ["ws"])
    assert out["ws_smean"].to_list() == [2.0, 15.0]
    assert out["ws_smin"].to_list() == [1.0, 10.0]
    assert out["ws_smax"].to_list() == [3.0, 20.0]
    # spatial std over {1,3} = 1.0 at t0
    assert abs(out["ws_sstd"].to_list()[0] - 1.0) < 1e-9


def test_aggregate_points_aligns_on_valid_time_intersection():
    a = _pt([1.0, 2.0])
    b = pl.DataFrame(
        {"valid_time": [dt.datetime(2024, 1, 1)], "ws": [5.0]},
        schema_overrides={"valid_time": pl.Datetime("us"), "ws": pl.Float64},
    )
    out = spatial.aggregate_points({0: a, 1: b}, ["ws"])
    assert out.height == 1  # only the shared timestamp survives
    assert out["ws_smean"].to_list() == [3.0]


def test_assign_rings_buckets_by_distance():
    rings = spatial.assign_rings([0.0, 0.05, 0.12, 0.2], n_rings=2, max_distance=0.2)
    assert rings[0] == 0 and rings[-1] == 1  # nearest in ring 0, farthest in ring 1
    assert len(rings) == 4


def _grid_frame(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
    })


def test_spatial_pca_fits_on_train_and_transforms_holdout():
    df = _grid_frame(50, 0)
    cols = ["a", "b", "c"]
    reducer = spatial.SpatialPCA(k=2).fit(df[:40], cols)
    z_full = reducer.transform(df)
    z_again = reducer.transform(df)
    assert z_full.shape == (50, 2)
    # deterministic: transforming the same rows twice matches
    assert (z_full.to_numpy() == z_again.to_numpy()).all()


def test_spatial_pca_refuses_unseen_columns():
    df = _grid_frame(20, 1)
    reducer = spatial.SpatialPCA(k=2).fit(df, ["a", "b", "c"])
    import pytest
    with pytest.raises(ValueError):
        reducer.transform(df.select(["a", "b"]))  # missing column 'c'
