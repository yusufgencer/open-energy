from __future__ import annotations

from dataclasses import dataclass

from openenergy.features.registry import available_blocks


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    kind: str
    blocks: tuple[str, ...]
    description: str


_FAMILIES = [
    FeatureFamily(
        name="wind_physical_core",
        kind="wind",
        blocks=("wind_power", "air_density", "wind_direction"),
        description="Rüzgar hızı güç dönüşümü, hava yoğunluğu ve yön.",
    ),
    FeatureFamily(
        name="wind_gust_shear",
        kind="wind",
        blocks=("wind_power", "wind_gust", "wind_shear", "wind_direction"),
        description="Gust, shear ve yön etkilerini birlikte dener.",
    ),
    FeatureFamily(
        name="wind_temporal_core",
        kind="wind",
        blocks=("wind_power", "cyclical_time"),
        description="Ana rüzgar sinyali + saat/mevsim döngüleri.",
    ),
    FeatureFamily(
        name="wind_trajectory_physics",
        kind="wind",
        blocks=("wind_power", "wind_physics_ext", "nwp_trajectory", "wind_direction"),
        description="Genişletilmiş rüzgar fiziği + NWP yörünge (lead) sinyalleri.",
    ),
    FeatureFamily(
        name="wind_spatial_fusion",
        kind="wind",
        blocks=("wind_power", "wind_direction", "wind_physics_ext", "nwp_trajectory"),
        description="Uzamsal grid politikalarıyla eşleşen rüzgar füzyon bundle'ı (yörünge dahil).",
    ),
    FeatureFamily(
        name="wind_full_tabular",
        kind="wind",
        blocks=tuple(available_blocks("wind")),
        description="Mevcut tüm rüzgar feature blokları.",
    ),
    FeatureFamily(
        name="solar_raw_irradiance",
        kind="solar",
        blocks=("gti", "cloud", "temperature_derating"),
        description="GTI, bulut ve sıcaklık derating sinyali.",
    ),
    FeatureFamily(
        name="solar_clear_sky",
        kind="solar",
        blocks=("clear_sky_index", "gti", "cloud", "temperature_derating"),
        description="Clear-sky index kullanan PV feature ailesi.",
    ),
    FeatureFamily(
        name="solar_cloud_temporal",
        kind="solar",
        blocks=("cloud", "cyclical_time", "temperature_derating"),
        description="Bulut rejimi ve temporal PV sinyalleri.",
    ),
    FeatureFamily(
        name="solar_spatial_cloud",
        kind="solar",
        blocks=("gti", "cloud", "nwp_trajectory", "temperature_derating"),
        description="Uzamsal bulut/GTI füzyonu + NWP yörünge (lead) sinyalleri.",
    ),
    FeatureFamily(
        name="solar_geometry_core",
        kind="solar",
        blocks=("solar_geometry", "gti", "cloud", "temperature_derating"),
        description="pvlib güneş geometrisi (cos_zenith vb.) + GTI/bulut/derating.",
    ),
    FeatureFamily(
        name="solar_physics_hybrid",
        kind="solar",
        blocks=("pv_model_chain", "solar_geometry", "gti", "cloud", "temperature_derating"),
        description="Fizik model-chain (P_physical) + güneş geometrisi ile hibrit PV ailesi.",
    ),
    FeatureFamily(
        name="solar_full_tabular",
        kind="solar",
        blocks=tuple(available_blocks("solar")),
        description="Mevcut tüm solar feature blokları.",
    ),
]

_BY_NAME = {family.name: family for family in _FAMILIES}


def available_families(kind: str) -> list[str]:
    return [family.name for family in _FAMILIES if family.kind == kind]


def get_family(name: str) -> FeatureFamily:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"bilinmeyen feature family: {name}") from exc


def blocks_for_family(name: str) -> list[str]:
    family = get_family(name)
    valid = set(available_blocks(family.kind))
    return [block for block in family.blocks if block in valid]
