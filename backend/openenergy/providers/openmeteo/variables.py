from __future__ import annotations

COMMON_VARS = ["temperature_2m", "surface_pressure", "relative_humidity_2m"]

WIND_VARS = [
    "wind_speed_10m", "wind_speed_100m", "wind_speed_120m",
    "wind_direction_100m", "wind_gusts_10m",
]

SOLAR_VARS = [
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
    "global_tilted_irradiance", "cloud_cover", "is_day",
    "terrestrial_solar_radiation", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
]


def default_variables(kind: str) -> list[str]:
    if kind == "wind":
        return WIND_VARS + COMMON_VARS
    if kind == "solar":
        return SOLAR_VARS + COMMON_VARS
    raise ValueError(f"bilinmeyen kind: {kind}")
