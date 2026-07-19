"""point_policy — how a grid of weather points collapses into model features.

The core new search axis of Phase B. A multi-point raw frame carries per-point
namespaced weather columns (`wind_speed_100m__p7`). Each policy turns those into
a canonical weather set (so the physics feature blocks, which look for canonical
names like `wind_speed_100m`, still work) plus optional spatial extras:

- single_best     — the centre point only (literature's negative control)
- spatial_mean    — mean across points (denoises; usually beats single point)
- mean_plus_spread— mean + cross-point std (`*_sstd`, an uncertainty signal)
- all_points      — mean canonical + every raw per-point column

Row order and the target column are never touched — only columns change, so the
runner's positional-index contract holds across policies.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

POINT_POLICIES = ["single_best", "spatial_mean", "mean_plus_spread", "all_points"]

_PRESERVE = ("valid_time", "power_mw", "quality_flagged", "quality_severe")
_NS_RE = re.compile(r"^(?P<base>.+)__p(?P<pid>\d+)$")


def _namespaced(columns: list[str]) -> dict[str, dict[int, str]]:
    groups: dict[str, dict[int, str]] = {}
    for c in columns:
        m = _NS_RE.match(c)
        if m:
            groups.setdefault(m.group("base"), {})[int(m.group("pid"))] = c
    return groups


def is_multipoint(raw: pl.DataFrame) -> bool:
    return bool(_namespaced(raw.columns))


def apply_point_policy(raw: pl.DataFrame, policy: str, *, point_ids: list[int]) -> pl.DataFrame:
    if policy not in POINT_POLICIES:
        raise ValueError(f"bilinmeyen point_policy: {policy}")
    groups = _namespaced(raw.columns)
    if not groups:
        return raw  # already single-point / canonical

    out = raw.select([c for c in _PRESERVE if c in raw.columns])
    center = point_ids[0]
    for base, by_pid in groups.items():
        cols = [by_pid[p] for p in point_ids if p in by_pid]
        if not cols:
            continue
        if policy == "single_best":
            src = by_pid.get(center, cols[0])
            out = out.with_columns(raw[src].alias(base))
            continue
        mat = raw.select(cols).to_numpy().astype(float)
        out = out.with_columns(pl.Series(base, np.nanmean(mat, axis=1)))
        if policy == "mean_plus_spread":
            out = out.with_columns(pl.Series(f"{base}_sstd", np.nanstd(mat, axis=1)))
        elif policy == "all_points":
            for p in point_ids:
                if p in by_pid:
                    out = out.with_columns(raw[by_pid[p]].alias(f"{base}__p{p}"))
    return out
