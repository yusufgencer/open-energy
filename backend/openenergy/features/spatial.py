"""Spatial aggregation over a grid of weather points.

Pure functions consumed by the point_policy stage (spatial_mean, mean+spread,
etc.). The literature (Andrade & Bessa 2017) shows spatial fusion — mean/std and
concentric-ring aggregates across surrounding grid points — beats picking the
single nearest point; spatial std doubles as an uncertainty-regime signal for the
quantile band. These helpers operate on per-point wide frames and are tested
standalone, before any real multi-point ingest exists.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

_STAT_SUFFIX = {
    "mean": "smean", "std": "sstd", "min": "smin",
    "max": "smax", "p25": "sp25", "p75": "sp75",
}
_DEFAULT_STATS = ("mean", "std", "min", "max", "p25", "p75")


def aggregate_points(
    frames: dict[int, pl.DataFrame],
    variables: list[str],
    stats: tuple[str, ...] = _DEFAULT_STATS,
) -> pl.DataFrame:
    """Cross-point statistics per timestamp.

    Each frame in `frames` (keyed by point_id) carries `valid_time` plus weather
    columns. Frames are inner-joined on valid_time (row universe = shared
    timestamps), then for each variable the requested stats are computed across
    the points, producing columns like `ws_smean`, `ws_sstd`. Returns a frame
    with valid_time + the aggregated columns.
    """
    pids = sorted(frames)
    merged: pl.DataFrame | None = None
    for pid in pids:
        f = frames[pid]
        keep = [v for v in variables if v in f.columns]
        sub = f.select(["valid_time", *keep]).rename({v: f"{v}__p{pid}" for v in keep})
        merged = sub if merged is None else merged.join(sub, on="valid_time", how="inner")
    if merged is None:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us")})

    out = merged.select("valid_time")
    for v in variables:
        cols = [f"{v}__p{pid}" for pid in pids if f"{v}__p{pid}" in merged.columns]
        if not cols:
            continue
        mat = merged.select(cols).to_numpy().astype(float)
        for stat in stats:
            if stat == "mean":
                vals = np.nanmean(mat, axis=1)
            elif stat == "std":
                vals = np.nanstd(mat, axis=1)
            elif stat == "min":
                vals = np.nanmin(mat, axis=1)
            elif stat == "max":
                vals = np.nanmax(mat, axis=1)
            elif stat == "p25":
                vals = np.nanpercentile(mat, 25, axis=1)
            elif stat == "p75":
                vals = np.nanpercentile(mat, 75, axis=1)
            else:
                raise ValueError(f"bilinmeyen stat: {stat}")
            out = out.with_columns(pl.Series(f"{v}_{_STAT_SUFFIX[stat]}", vals))
    return out


class SpatialPCA:
    """PCA over grid×variable columns, fit on training rows only.

    The best spatial method for wind in the literature. It MUST be fit on the
    training slice inside a fold and applied unchanged to val/test/serving —
    fitting on all rows would leak. The fitted reducer is stored in the strategy
    artifact so serving reproduces the exact projection. `transform` refuses any
    frame missing a column it was fit on.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self._pca: PCA | None = None
        self._cols: list[str] | None = None

    def fit(self, df: pl.DataFrame, columns: list[str]) -> "SpatialPCA":
        self._cols = list(columns)
        X = np.nan_to_num(df.select(self._cols).to_numpy().astype(float))
        n_comp = min(self.k, X.shape[1], max(X.shape[0], 1))
        self._pca = PCA(n_components=n_comp, random_state=42).fit(X)
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._pca is None or self._cols is None:
            raise ValueError("SpatialPCA.transform önce fit() gerektirir")
        missing = [c for c in self._cols if c not in df.columns]
        if missing:
            raise ValueError(f"SpatialPCA.transform eksik kolonlar: {missing}")
        X = np.nan_to_num(df.select(self._cols).to_numpy().astype(float))
        Z = self._pca.transform(X)
        return pl.DataFrame({f"spca_{i}": Z[:, i] for i in range(Z.shape[1])})


def assign_rings(distances: list[float], n_rings: int, max_distance: float) -> list[int]:
    """Bucket each point's distance-from-centre into one of n_rings equal-width rings.

    Ring 0 is nearest the plant, ring n_rings-1 the outermost. Distances at or
    beyond max_distance clamp to the outer ring.
    """
    if n_rings < 1:
        raise ValueError("n_rings >= 1 olmalı")
    width = max_distance / n_rings
    out = []
    for d in distances:
        idx = int(d / width) if width > 0 else 0
        out.append(min(max(idx, 0), n_rings - 1))
    return out
