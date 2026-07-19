from __future__ import annotations

import inspect

import polars as pl
from pydantic import BaseModel, field_validator

from openenergy.features.registry import REGISTRY, available_blocks

# Context values threaded from FeatureConfig into blocks that declare a matching
# keyword parameter (e.g. solar_geometry needs plant lat/lon). Blocks that do not
# declare a parameter are called unchanged, so other blocks stay unaffected.
_CONTEXT_FIELDS = ("lat", "lon")


class FeatureConfig(BaseModel):
    blocks: list[str]
    lat: float | None = None
    lon: float | None = None

    @field_validator("blocks")
    @classmethod
    def no_duplicates(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for name in v:
            if name in seen:
                raise ValueError(f"duplicate block name: '{name}'")
            seen.add(name)
        return v


def build_features(df: pl.DataFrame, kind: str, config: FeatureConfig) -> pl.DataFrame:
    allowed = set(available_blocks(kind))
    context = {field: getattr(config, field) for field in _CONTEXT_FIELDS}
    out = df
    for name in config.blocks:
        if name not in REGISTRY:
            raise ValueError(f"bilinmeyen block: {name}")
        if name not in allowed:
            raise ValueError(f"'{name}' block'u '{kind}' tipine uygulanamaz")
        func = REGISTRY[name].func
        params = inspect.signature(func).parameters
        kwargs = {field: value for field, value in context.items() if field in params}
        out = func(out, **kwargs)
    return out


def default_config(kind: str) -> FeatureConfig:
    return FeatureConfig(blocks=available_blocks(kind))
