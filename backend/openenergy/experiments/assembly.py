from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

from openenergy.datasets.horizon import build_horizon_dataset, build_multipoint_dataset
from openenergy.features import FeatureConfig, build_features
from openenergy.features.point_policy import apply_point_policy, is_multipoint
from openenergy.features.target import PlantGeometry, kpv_forward, normalize_power
from openenergy.physics.solar import clear_sky_power
from openenergy.physics.solar import daylight_mask as _solar_daylight_mask

_EXCLUDE = {
    "power_mw", "power_norm", "valid_time", "quality_flagged", "quality_severe",
}


@dataclass
class AssembledDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    valid_time: np.ndarray
    # kPV bookkeeping (C1-t3). ``p_cs`` is the clear-sky envelope aligned to the
    # (valid_time-sorted) rows; only populated when target_policy == "kpv".
    # ``target_policy`` records which target space ``y`` lives in so serving can
    # apply the matching inverse transform. Defaults keep the legacy
    # capacity_norm path byte-identical (extra fields, positional ctor unaffected).
    p_cs: np.ndarray | None = None
    target_policy: str = "capacity_norm"
    # True means the target is admissible for clean training/evaluation.  It is
    # bookkeeping only: raw assembly never drops flagged rows and split indices
    # are always computed over the complete row universe.
    clean_mask: np.ndarray | None = None


def assemble_raw(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                 horizon_hours: int, source: str | None = None,
                 variables: list[str] | None = None) -> pl.DataFrame:
    """Return the raw dataset sorted by valid_time (fixed row set: valid_time + raw weather + power_mw).

    This is the stable row universe over which positional indices are computed.
    Feature config changes COLUMNS only — row order never changes.
    ``source`` optionally restricts weather_raw.model to a single NWP source.
    """
    raw = build_horizon_dataset(con, plant_id=plant_id, point_id=point_id,
                                horizon_hours=horizon_hours, source=source,
                                variables=variables)
    if raw.height == 0:
        return raw
    return raw.sort("valid_time")


def featurize(raw: pl.DataFrame, *, kind: str, feature_config: FeatureConfig,
              capacity_mw: float, point_policy: str = "single_point",
              point_ids: list[int] | None = None,
              target_policy: str = "capacity_norm",
              geometry: PlantGeometry | None = None,
              daylight_mask: bool = False) -> AssembledDataset:
    """Apply feature engineering and target normalisation to a raw frame.

    Toggling feature_config.blocks changes which engineered columns are added, so X's
    column count genuinely reflects the chosen blocks. Raw weather columns always remain.

    When ``raw`` carries per-point namespaced columns (a grid) and point_policy is not
    the legacy "single_point", the grid is first collapsed via apply_point_policy so the
    feature blocks see canonical weather columns. Row order is untouched — only columns
    change — so the runner's positional-index contract holds across policies.

    ``target_policy`` selects the training target space (C1-t3):

    - ``capacity_norm`` (default, legacy): ``y = P / capacity`` — byte-identical to before.
    - ``kpv``: ``y' = P / P_cs`` (clear-sky ratio), where ``P_cs`` is rebuilt from the
      plant ``geometry`` over the sorted valid_time. The envelope ``p_cs`` is returned so
      serving can invert ``ŷ' × P_cs``. kPV requires ``geometry``.

    ``daylight_mask`` (C1-t4, solar only): when ``True`` the night / near-horizon
    rows (apparent zenith > ~85°) are dropped so a solar model never trains on
    zero-noise dark hours. Requires ``geometry`` (for lat/lon). It is a *separate,
    opt-in* assembly variant — the default (``False``) path keeps the full fixed
    row universe, so the runner's positional train/test-index contract is
    untouched. For wind the flag is a no-op (wind is unaffected).
    """
    if raw.height == 0:
        return AssembledDataset(np.empty((0, 0)), np.empty((0,)), [], np.empty((0,)))
    if point_policy != "single_point" and is_multipoint(raw):
        pids = point_ids or sorted(
            {int(c.rsplit("__p", 1)[1]) for c in raw.columns if "__p" in c}
        )
        raw = apply_point_policy(raw, point_policy, point_ids=pids)
    feats = build_features(raw, kind, feature_config)
    feats = normalize_power(feats, capacity_mw)
    feats = feats.sort("valid_time")
    feature_names = [c for c in feats.columns
                     if c not in _EXCLUDE and feats.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    X = feats.select(feature_names).to_numpy()
    vt = feats["valid_time"].to_numpy()

    p_cs = None
    if target_policy == "kpv":
        if geometry is None:
            raise ValueError("target_policy='kpv' için geometry gerekli")
        p_cs = clear_sky_power(
            feats["valid_time"].to_list(),
            geometry.lat, geometry.lon, geometry.tilt, geometry.azimuth, capacity_mw,
        ).to_numpy()
        y = kpv_forward(feats["power_mw"].to_numpy(), p_cs)
    elif target_policy == "capacity_norm":
        y = feats["power_norm"].to_numpy()
    else:
        raise ValueError(f"bilinmeyen target_policy: {target_policy}")

    if daylight_mask and kind == "solar":
        if geometry is None:
            raise ValueError("daylight_mask=True için geometry gerekli")
        keep = _solar_daylight_mask(feats["valid_time"].to_list(), geometry.lat, geometry.lon)
        X = X[keep]
        y = y[keep]
        vt = vt[keep]
        if p_cs is not None:
            p_cs = p_cs[keep]

    clean_mask = (
        ~feats["quality_severe"].fill_null(False).to_numpy()
        if "quality_severe" in feats.columns
        else np.ones(feats.height, dtype=bool)
    )
    if daylight_mask and kind == "solar":
        clean_mask = clean_mask[keep]

    return AssembledDataset(X=X, y=y, feature_names=feature_names, valid_time=vt,
                            p_cs=p_cs, target_policy=target_policy,
                            clean_mask=np.asarray(clean_mask, dtype=bool))


def assemble(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int, horizon_hours: int,
             capacity_mw: float, kind: str, feature_config: FeatureConfig,
             variables: list[str] | None = None) -> AssembledDataset:
    return featurize(
        assemble_raw(con, plant_id=plant_id, point_id=point_id,
                     horizon_hours=horizon_hours, variables=variables),
        kind=kind, feature_config=feature_config, capacity_mw=capacity_mw,
    )


@dataclass
class PooledDataset:
    """One dataset pooled across leads for the F1-t2 pooled-GBM competitor.

    ``X`` carries the usual feature columns PLUS a final ``lead_time_hours``
    feature so a single model learns how skill/spread change with lead. ``y`` is
    normalized power. ``valid_time`` and ``lead_time_hours`` are row-aligned
    bookkeeping arrays: ``valid_time`` drives leakage-safe group splits (the same
    time recurs once per lead), ``lead_time_hours`` drives per-lead-bucket
    calibration and per-lead metric slicing. Rows are ordered by
    ``(valid_time, lead)`` — a stable pooled universe.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    valid_time: np.ndarray
    lead_time_hours: np.ndarray


def assemble_pooled(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                    horizon_hours_list: list[int], capacity_mw: float, kind: str,
                    feature_config: FeatureConfig, source: str | None = None,
                    variables: list[str] | None = None) -> PooledDataset:
    """Assemble a leak-safe pooled-over-leads dataset (F1-t2).

    Each lead is assembled and featurized *independently* (so per-lead feature
    engineering never mixes rows from different leads), then the per-lead feature
    matrices are stacked and a single ``lead_time_hours`` feature column is
    appended. Feature columns are identical across leads (same kind/blocks), so
    the stack is well-formed. The single-lead (legacy) assembly path is untouched.
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    vts: list[np.ndarray] = []
    leads: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for lead in sorted(dict.fromkeys(horizon_hours_list)):
        raw = assemble_raw(con, plant_id=plant_id, point_id=point_id,
                           horizon_hours=lead, source=source, variables=variables)
        if raw.height == 0:
            continue
        ds = featurize(raw, kind=kind, feature_config=feature_config, capacity_mw=capacity_mw)
        if ds.X.shape[0] == 0:
            continue
        feature_names = ds.feature_names
        Xs.append(ds.X)
        ys.append(ds.y)
        vts.append(ds.valid_time)
        leads.append(np.full(ds.X.shape[0], float(lead), dtype=float))
    if feature_names is None:
        return PooledDataset(np.empty((0, 0)), np.empty((0,)), [], np.empty((0,)), np.empty((0,)))
    X = np.vstack(Xs)
    lead_arr = np.concatenate(leads)
    X = np.column_stack([X, lead_arr])  # lead_time_hours as the final feature
    y = np.concatenate(ys)
    valid_time = np.concatenate(vts)
    # Stable pooled universe: sort by (valid_time, lead). lexsort's last key is
    # primary, so keys=(lead, valid_time) sorts by valid_time then lead.
    order = np.lexsort((lead_arr, valid_time))
    return PooledDataset(
        X=X[order], y=y[order], feature_names=feature_names + ["lead_time_hours"],
        valid_time=valid_time[order], lead_time_hours=lead_arr[order],
    )
