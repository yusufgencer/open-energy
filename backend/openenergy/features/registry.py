from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from openenergy.features import blocks


@dataclass(frozen=True)
class Block:
    name: str
    func: Callable[[pl.DataFrame], pl.DataFrame]
    kinds: tuple[str, ...]


_ALL = [
    Block("wind_power", blocks.wind_power, ("wind",)),
    Block("air_density", blocks.air_density, ("wind",)),
    Block("wind_direction", blocks.wind_direction, ("wind",)),
    Block("wind_shear", blocks.wind_shear, ("wind",)),
    Block("wind_gust", blocks.wind_gust, ("wind",)),
    Block("wind_physics_ext", blocks.wind_physics_ext, ("wind",)),
    Block("clear_sky_index", blocks.clear_sky_index, ("solar",)),
    Block("gti", blocks.gti, ("solar",)),
    Block("cloud", blocks.cloud, ("solar",)),
    Block("temperature_derating", blocks.temperature_derating, ("solar",)),
    Block("solar_geometry", blocks.solar_geometry, ("solar",)),
    Block("pv_model_chain", blocks.pv_model_chain, ("solar",)),
    Block("cyclical_time", blocks.cyclical_time, ("wind", "solar")),
    Block("nwp_trajectory", blocks.nwp_trajectory, ("wind", "solar")),
]

REGISTRY: dict[str, Block] = {b.name: b for b in _ALL}


def available_blocks(kind: str) -> list[str]:
    return [b.name for b in _ALL if kind in b.kinds]


def get_block(name: str) -> Block:
    if name not in REGISTRY:
        raise ValueError(f"bilinmeyen block: {name}")
    return REGISTRY[name]
