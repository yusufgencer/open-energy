from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Available target transforms (C1-t3). ``capacity_norm`` is the legacy path:
# train on P/capacity, serve ŷ×capacity. ``kpv`` trains on the clear-sky ratio
# P/P_cs and serves ŷ'×P_cs — physics supplies the peak envelope so trees never
# have to extrapolate the diurnal/seasonal amplitude.
TARGET_POLICIES = ("capacity_norm", "kpv")

# Ceiling on the clear-sky ratio y' = P/P_cs. Real production can briefly exceed
# the modelled clear-sky envelope (cloud enhancement, model bias); clip so a
# handful of >1 ratios cannot dominate the fit, but leave head-room above 1.
_KPV_CLIP_MAX = 1.5

# P_cs values at/below this (MW) are treated as night: no division, target 0.
_PCS_FLOOR = 1e-6


@dataclass(frozen=True)
class PlantGeometry:
    """Fixed-tilt PV geometry needed to reconstruct the clear-sky envelope P_cs.

    Stored on the strategy artifact so serving can rebuild the exact denominator
    used at training time (the "geometry ref" of the kPV policy).
    """

    lat: float
    lon: float
    tilt: float
    azimuth: float

    def as_dict(self) -> dict[str, float]:
        return {"lat": self.lat, "lon": self.lon, "tilt": self.tilt, "azimuth": self.azimuth}


def normalize_power(df: pl.DataFrame, capacity_mw: float, *,
                    source: str = "power_mw", out: str = "power_norm") -> pl.DataFrame:
    if capacity_mw <= 0:
        raise ValueError("capacity_mw > 0 olmalı")
    if source not in df.columns:
        raise ValueError(f"'{source}' kolonu yok")
    return df.with_columns((pl.col(source) / capacity_mw).clip(0.0, 1.2).cast(pl.Float64).alias(out))


def kpv_forward(power_mw, p_cs, *, clip_max: float = _KPV_CLIP_MAX,
                floor: float = _PCS_FLOOR) -> np.ndarray:
    """Forward kPV transform: y' = P / P_cs on daylight rows, 0 at night.

    Night rows (``p_cs <= floor``) never divide — their ratio is defined as 0.
    Daylight ratios are clipped to ``[0, clip_max]`` so cloud-enhancement spikes
    above the clear-sky envelope cannot dominate the fit.
    """
    power = np.asarray(power_mw, dtype="float64")
    pcs = np.asarray(p_cs, dtype="float64")
    y = np.zeros_like(power)
    day = pcs > floor
    y[day] = power[day] / pcs[day]
    return np.clip(y, 0.0, clip_max)


def kpv_inverse(y_prime, p_cs, capacity_mw: float) -> np.ndarray:
    """Inverse kPV transform: ŷ = clip(ŷ' × P_cs, 0, capacity).

    Multiplies the predicted clear-sky ratio back by the physical envelope and
    clamps to the plant nameplate so served power is always in ``[0, capacity]``.
    """
    yp = np.asarray(y_prime, dtype="float64")
    pcs = np.asarray(p_cs, dtype="float64")
    return np.clip(yp * pcs, 0.0, capacity_mw)
