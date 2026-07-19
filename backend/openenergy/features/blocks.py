from __future__ import annotations

import math

import polars as pl

PRESERVE_COLS = ["valid_time", "power_mw"]
EPS = 1e-9


def _first_present(df: pl.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Wind feature blocks
# ---------------------------------------------------------------------------

def wind_power(df: pl.DataFrame) -> pl.DataFrame:
    col = _first_present(df, ["wind_speed_120m", "wind_speed_100m", "wind_speed_80m", "wind_speed_10m"])
    if col is None:
        return df
    return df.with_columns(
        pl.col(col).cast(pl.Float64).alias("wind_speed"),
        (pl.col(col) ** 2).cast(pl.Float64).alias("wind_speed_sq"),
        (pl.col(col) ** 3).cast(pl.Float64).alias("wind_speed_cube"),
    )


def air_density(df: pl.DataFrame) -> pl.DataFrame:
    if "surface_pressure" in df.columns and "temperature_2m" in df.columns:
        df = df.with_columns(
            (pl.col("surface_pressure") * 100.0
             / (287.05 * (pl.col("temperature_2m") + 273.15))).cast(pl.Float64).alias("air_density")
        )
    if "air_density" in df.columns and "wind_speed_cube" in df.columns:
        df = df.with_columns(
            (pl.col("air_density") * pl.col("wind_speed_cube")).cast(pl.Float64).alias("rho_v3")
        )
    return df


def wind_direction(df: pl.DataFrame) -> pl.DataFrame:
    col = _first_present(df, ["wind_direction_120m", "wind_direction_100m", "wind_direction_80m", "wind_direction_10m"])
    if col is None:
        return df
    rad = pl.col(col) * (math.pi / 180.0)
    return df.with_columns(
        rad.sin().cast(pl.Float64).alias("wind_dir_sin"),
        rad.cos().cast(pl.Float64).alias("wind_dir_cos"),
    )


def wind_shear(df: pl.DataFrame) -> pl.DataFrame:
    if "wind_speed_100m" not in df.columns or "wind_speed_10m" not in df.columns:
        return df
    return df.with_columns(
        (pl.col("wind_speed_100m") / (pl.col("wind_speed_10m") + EPS)).cast(pl.Float64).alias("wind_shear")
    )


_WS_COLS = ["wind_speed_100m", "wind_speed_120m", "wind_speed_80m", "wind_speed_10m"]
_WD_COLS = ["wind_direction_100m", "wind_direction_120m", "wind_direction_80m", "wind_direction_10m"]


def wind_physics_ext(df: pl.DataFrame, *, cut_in: float = 3.0, rated: float = 12.0,
                     cut_out: float = 25.0) -> pl.DataFrame:
    """Generic power-curve transform, cut-out flag and u/v wind components.

    pc(v) = clip((v³ - v_ci³)/(v_r³ - v_ci³), 0, 1), forced to 0 above cut-out;
    a cut-out indicator; and u = -v·sin(dir), v = -v·cos(dir) when a direction
    column is present. These linearise the target for the trees. No-op without
    a wind-speed column; u/v skipped when no direction column exists.
    """
    ws = _first_present(df, _WS_COLS)
    if ws is None:
        return df
    v = pl.col(ws).cast(pl.Float64)
    pc = ((v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)).clip(0.0, 1.0)
    pc = pl.when(v > cut_out).then(pl.lit(0.0)).otherwise(pc)
    df = df.with_columns(
        pc.cast(pl.Float64).alias("wind_pc"),
        (v > cut_out).cast(pl.Float64).alias("wind_cutout"),
    )
    wd = _first_present(df, _WD_COLS)
    if wd is not None:
        rad = pl.col(wd) * (math.pi / 180.0)
        df = df.with_columns(
            (-v * rad.sin()).cast(pl.Float64).alias("wind_u"),
            (-v * rad.cos()).cast(pl.Float64).alias("wind_v"),
        )
    return df


def wind_gust(df: pl.DataFrame) -> pl.DataFrame:
    if "wind_gusts_10m" not in df.columns:
        return df
    if "wind_speed" in df.columns:
        df = df.with_columns(
            (pl.col("wind_gusts_10m") / (pl.col("wind_speed") + EPS)).cast(pl.Float64).alias("gust_factor")
        )
    return df


# ---------------------------------------------------------------------------
# Solar feature blocks
# ---------------------------------------------------------------------------

def clear_sky_index(df: pl.DataFrame) -> pl.DataFrame:
    if "shortwave_radiation" not in df.columns or "terrestrial_solar_radiation" not in df.columns:
        return df
    kt = pl.when(pl.col("terrestrial_solar_radiation") <= 0.0).then(pl.lit(0.0)).otherwise(
        (pl.col("shortwave_radiation") / pl.col("terrestrial_solar_radiation")).clip(0.0, 1.5)
    )
    return df.with_columns(kt.cast(pl.Float64).alias("clear_sky_index"))


def gti(df: pl.DataFrame) -> pl.DataFrame:
    if "global_tilted_irradiance" not in df.columns:
        return df
    return df.with_columns(pl.col("global_tilted_irradiance").cast(pl.Float64).alias("gti"))


def cloud(df: pl.DataFrame) -> pl.DataFrame:
    present = [c for c in ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"] if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in present])


def temperature_derating(df: pl.DataFrame) -> pl.DataFrame:
    irr = _first_present(df, ["global_tilted_irradiance", "shortwave_radiation"])
    if "temperature_2m" not in df.columns or irr is None:
        return df
    df = df.with_columns((pl.col("temperature_2m") + 0.035 * pl.col(irr)).cast(pl.Float64).alias("cell_temp"))
    return df.with_columns((1.0 - 0.004 * (pl.col("cell_temp") - 25.0)).cast(pl.Float64).alias("pv_derate"))


_MC_WIND_COLS = ["wind_speed_10m", "wind_speed_100m", "wind_speed_120m", "wind_speed_80m"]


def pv_model_chain(df: pl.DataFrame) -> pl.DataFrame:
    """Physics model-chain feature ``p_physical`` (Mayer 2022 hybrid pattern).

    Runs the pvlib chain GTI -> Faiman cell temperature -> PVWatts DC power on
    the plane-of-array irradiance column (``global_tilted_irradiance``) and adds
    the resulting unit-scaled physical power as a single feature the ML model can
    anchor on. Ambient temperature (``temperature_2m``) and wind speed feed the
    Faiman cell-temp step when present; otherwise physical defaults are used.

    No-op (changes no columns) without a GTI column. Deterministic, leakage-free
    (all inputs are the forecast NWP fields) and column-only: row order/count are
    preserved so it stays aligned with the valid_time-sorted frame.
    """
    if "global_tilted_irradiance" not in df.columns:
        return df

    from openenergy.physics.solar import pv_model_chain_power

    gti = df["global_tilted_irradiance"].cast(pl.Float64).to_numpy()
    temp = (
        df["temperature_2m"].cast(pl.Float64).to_numpy()
        if "temperature_2m" in df.columns
        else None
    )
    ws_col = _first_present(df, _MC_WIND_COLS)
    wind = df[ws_col].cast(pl.Float64).to_numpy() if ws_col is not None else None

    power = pv_model_chain_power(gti, temp, wind)
    return df.with_columns(pl.Series("p_physical", power).cast(pl.Float64))


def solar_geometry(df: pl.DataFrame, *, lat: float | None = None,
                   lon: float | None = None) -> pl.DataFrame:
    """Sun-geometry features (cos_zenith, elevation, azimuth sin/cos, airmass,
    extraterrestrial DNI, day length, hours since sunrise / to sunset, daylight
    flag) computed via pvlib from ``valid_time`` and the plant coordinates.

    Coordinates are threaded in from ``FeatureConfig`` by the pipeline. The block
    no-ops (changes no columns) when either coordinate is missing or ``valid_time``
    is not an actual Datetime column, so other blocks and the coord-free path are
    unaffected. Purely deterministic and leakage-free at any forecast horizon.
    """
    if lat is None or lon is None:
        return df
    if "valid_time" not in df.columns:
        return df
    if not isinstance(df["valid_time"].dtype, pl.Datetime):
        return df

    from openenergy.physics.solar import solar_geometry_features

    geo = solar_geometry_features(df["valid_time"].to_list(), lat, lon)
    return df.with_columns(
        [pl.Series(name, values).cast(pl.Float64) for name, values in geo.items()]
    )


# ---------------------------------------------------------------------------
# NWP trajectory block (lag/lead + rolling context of the forecast series)
# ---------------------------------------------------------------------------

_TRAJ_DRIVERS = [
    "wind_speed_100m", "wind_speed_120m", "wind_speed_80m",
    "shortwave_radiation", "global_tilted_irradiance",
]


def nwp_trajectory(df: pl.DataFrame) -> pl.DataFrame:
    """Neighbouring-timestep context of the leading NWP driver.

    Adds t±1/±2/±3 lag & lead columns plus centered 3-step rolling mean/std/
    min/max. Leakage-safe: the whole forecast trajectory is available at issue
    time, so future (lead) values of the *forecast* are known. Operates on the
    valid_time-sorted frame; boundary rows get nulls (models nan-fill them).
    No-op when no known driver column is present.
    """
    col = _first_present(df, _TRAJ_DRIVERS)
    if col is None:
        return df
    c = pl.col(col).cast(pl.Float64)
    exprs = []
    for k in (1, 2, 3):
        exprs.append(c.shift(k).alias(f"{col}_lag{k}"))
        exprs.append(c.shift(-k).alias(f"{col}_lead{k}"))
    exprs.append(c.rolling_mean(window_size=3, center=True).alias(f"{col}_roll3_mean"))
    exprs.append(c.rolling_std(window_size=3, center=True).alias(f"{col}_roll3_std"))
    exprs.append(c.rolling_min(window_size=3, center=True).alias(f"{col}_roll3_min"))
    exprs.append(c.rolling_max(window_size=3, center=True).alias(f"{col}_roll3_max"))
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Cyclical time block
# ---------------------------------------------------------------------------

def cyclical_time(df: pl.DataFrame, time_col: str = "valid_time") -> pl.DataFrame:
    if time_col not in df.columns:
        return df
    # Only proceed if the column is an actual Datetime type (not Null / unknown)
    if not isinstance(df[time_col].dtype, pl.Datetime):
        return df
    two_pi = 2 * math.pi
    hour = pl.col(time_col).dt.hour().cast(pl.Float64)
    doy = pl.col(time_col).dt.ordinal_day().cast(pl.Float64)
    month = pl.col(time_col).dt.month().cast(pl.Float64)
    return df.with_columns(
        (two_pi * hour / 24.0).sin().alias("hour_sin"),
        (two_pi * hour / 24.0).cos().alias("hour_cos"),
        (two_pi * doy / 365.25).sin().alias("doy_sin"),
        (two_pi * doy / 365.25).cos().alias("doy_cos"),
        (two_pi * month / 12.0).sin().alias("month_sin"),
        (two_pi * month / 12.0).cos().alias("month_cos"),
    )
