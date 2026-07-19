"""Deterministic clear-sky PV power envelope.

``clear_sky_power`` produces ``P_cs(t)`` — the clear-sky (upper-envelope)
power a plant would generate at each timestamp — using pvlib's Ineichen-Perez
clear-sky model transposed onto the plant's plane-of-array and mapped to power
with a PVWatts-style linear envelope.

P_cs is the denominator of the kPV target transform (C1-t3), the source of the
P_cs-proportional sample weights (C1-t5) and the daylight mask (C1-t4). It is
purely a function of location, geometry, capacity and time — deterministic and
leakage-free at any forecast horizon (it never touches observed weather).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

# Reference plane-of-array irradiance for the PVWatts-style envelope (W/m^2).
_REFERENCE_POA = 1000.0


def _as_utc_index(times: Sequence | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce ``times`` to a tz-aware (UTC) ``DatetimeIndex``.

    Naive timestamps are interpreted as UTC; tz-aware inputs are converted.
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def clear_sky_power(
    times: Sequence | pd.DatetimeIndex,
    lat: float,
    lon: float,
    tilt: float,
    azimuth: float,
    capacity_mw: float,
) -> pd.Series:
    """Clear-sky power envelope ``P_cs`` (MW) for a fixed-tilt plant.

    Parameters
    ----------
    times:
        Timestamps to evaluate. A tz-naive input is treated as UTC.
    lat, lon:
        Plant latitude / longitude in decimal degrees.
    tilt:
        Surface tilt from horizontal, degrees (0 = flat).
    azimuth:
        Surface azimuth, degrees clockwise from north (180 = due south).
    capacity_mw:
        Plant nameplate capacity in MW; the returned series is clipped to it.

    Returns
    -------
    pandas.Series
        Power in MW indexed by the (UTC) input timestamps. ``0`` at night,
        rising to a peak near solar noon, always within ``[0, capacity_mw]``.
    """
    if capacity_mw < 0:
        raise ValueError("capacity_mw must be non-negative")

    index = _as_utc_index(times)
    if len(index) == 0:
        return pd.Series([], index=index, dtype="float64", name="p_cs")

    location = Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(index)
    clear_sky = location.get_clearsky(index, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    # PVWatts-style linear envelope: power tracks POA irradiance, saturating at
    # nameplate capacity. No temperature derate here — this is the clear-sky
    # ceiling, not an operating-point estimate.
    power = capacity_mw * (poa_global / _REFERENCE_POA)
    power = power.clip(lower=0.0, upper=capacity_mw)

    # Force sub-horizon / below-horizon sun to exactly zero.
    night = solar_position["apparent_elevation"] <= 0.0
    power = power.where(~night, 0.0)

    power = power.astype("float64")
    power.index = index
    power.name = "p_cs"
    return power


def daylight_mask(
    times: Sequence | pd.DatetimeIndex,
    lat: float,
    lon: float,
    *,
    max_zenith: float = 85.0,
) -> np.ndarray:
    """Boolean daylight mask, ``True`` where the sun is usefully above the horizon.

    A row is kept (``True``) when the apparent solar zenith is ``<= max_zenith``;
    night / near-horizon rows (zenith ``> max_zenith``) are ``False``. These
    excluded rows carry no PV signal — clear-sky power is ~0 there — so they only
    inject zero-noise into a solar model (C1-t4). Used to drop night rows from
    solar training (assembly) and to force a hard night zero in serving.

    Deterministic and leakage-free: a pure function of location and time. Times
    are interpreted as UTC when tz-naive. Row order is preserved exactly, so the
    mask aligns positionally with the valid_time-sorted feature frame.
    """
    index = _as_utc_index(times)
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    location = Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(index)
    zenith = solar_position["apparent_zenith"].to_numpy(dtype="float64")
    return zenith <= max_zenith


# Default PVWatts parameters for the model-chain feature. ``pdc0`` is normalized
# to 1.0 so the output is a unit-scaled physical-power feature (proportional to
# real production), not an absolute MW estimate; ``gamma_pdc`` is a typical
# crystalline-silicon temperature coefficient (per degC).
_MODEL_CHAIN_PDC0 = 1.0
_MODEL_CHAIN_GAMMA = -0.004


def pv_model_chain_power(
    gti,
    temp_air=None,
    wind_speed=None,
    *,
    pdc0: float = _MODEL_CHAIN_PDC0,
    gamma_pdc: float = _MODEL_CHAIN_GAMMA,
) -> np.ndarray:
    """Physics model-chain PV power ``P_physical`` from plane-of-array irradiance.

    Runs the pvlib hybrid chain used as a *feature* (Mayer 2022): GTI ->
    Faiman cell temperature -> PVWatts DC power. The result is a unit-scaled
    (``pdc0=1``) physical-power estimate proportional to real production, giving
    an ML model a strong physics anchor without extrapolating the peak.

    Parameters
    ----------
    gti:
        Global tilted (plane-of-array) irradiance, W/m^2. The only required
        driver; ``NaN`` or negative values yield zero power.
    temp_air:
        Ambient air temperature, degC. Defaults to 20 degC when ``None``.
    wind_speed:
        Wind speed at module height, m/s. Defaults to pvlib's Faiman default
        (1 m/s) when ``None``.
    pdc0, gamma_pdc:
        PVWatts nameplate (normalized) and temperature coefficient.

    Returns
    -------
    numpy.ndarray
        Non-negative float64 power aligned element-wise to ``gti``; exactly 0
        where irradiance is 0/negative/NaN (so night is always zero). Row order
        and count are preserved.
    """
    gti = np.asarray(gti, dtype="float64")
    n = gti.size
    if n == 0:
        return np.zeros(0, dtype="float64")

    gti_clean = np.nan_to_num(gti, nan=0.0)
    gti_clean = np.clip(gti_clean, 0.0, None)

    if temp_air is None:
        temp = np.full(n, 20.0, dtype="float64")
    else:
        temp = np.nan_to_num(np.asarray(temp_air, dtype="float64"), nan=20.0)
    if wind_speed is None:
        wind = np.full(n, 1.0, dtype="float64")
    else:
        wind = np.nan_to_num(np.asarray(wind_speed, dtype="float64"), nan=1.0)

    cell_temp = pvlib.temperature.faiman(gti_clean, temp, wind)
    power = pvlib.pvsystem.pvwatts_dc(gti_clean, cell_temp, pdc0, gamma_pdc)
    power = np.asarray(power, dtype="float64")
    # PVWatts can go slightly negative for hot, low-irradiance cells; the
    # physical envelope is non-negative and exactly zero without irradiance.
    power = np.clip(power, 0.0, None)
    power[gti_clean <= 0.0] = 0.0
    return power


# Feature columns produced by ``solar_geometry_features`` / the solar_geometry block.
SOLAR_GEOMETRY_COLUMNS = (
    "cos_zenith",
    "solar_elevation",
    "azimuth_sin",
    "azimuth_cos",
    "airmass",
    "extra_radiation",
    "day_length",
    "hours_since_sunrise",
    "hours_to_sunset",
    "is_daylight",
)


def solar_geometry_features(
    times: Sequence | pd.DatetimeIndex,
    lat: float,
    lon: float,
) -> dict[str, np.ndarray]:
    """Sun-geometry features for a location, aligned to ``times`` order.

    Returns one float64 array per key in :data:`SOLAR_GEOMETRY_COLUMNS`:

    - ``cos_zenith`` — cosine of the apparent solar zenith (the single
      strongest solar predictor); negative when the sun is below the horizon.
    - ``solar_elevation`` — apparent solar elevation, degrees (<= 0 at night).
    - ``azimuth_sin`` / ``azimuth_cos`` — unit-circle encoding of solar azimuth.
    - ``airmass`` — relative optical airmass (Kasten-Young); 0 at night.
    - ``extra_radiation`` — extraterrestrial (top-of-atmosphere) DNI, W/m^2.
    - ``day_length`` — daylight duration for the row's date, hours.
    - ``hours_since_sunrise`` / ``hours_to_sunset`` — signed offsets, hours
      (negative before sunrise / after sunset).
    - ``is_daylight`` — 1.0 when the sun is above the horizon, else 0.0.

    Deterministic and leakage-free: a pure function of location and time. Times
    are interpreted as UTC when tz-naive. Row order is preserved exactly, so the
    result can be attached column-wise to the valid_time-sorted feature frame.
    """
    index = _as_utc_index(times)
    n = len(index)
    if n == 0:
        empty = np.empty(0, dtype="float64")
        return {col: empty.copy() for col in SOLAR_GEOMETRY_COLUMNS}

    location = Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(index)
    zenith = solar_position["apparent_zenith"].to_numpy(dtype="float64")
    elevation = solar_position["apparent_elevation"].to_numpy(dtype="float64")
    azimuth = np.radians(solar_position["azimuth"].to_numpy(dtype="float64"))

    cos_zenith = np.cos(np.radians(zenith))
    airmass = pvlib.atmosphere.get_relative_airmass(solar_position["apparent_zenith"])
    airmass = np.nan_to_num(airmass.to_numpy(dtype="float64"), nan=0.0)
    extra = pvlib.irradiance.get_extra_radiation(index).to_numpy(dtype="float64")

    rise_set = location.get_sun_rise_set_transit(index)
    sunrise = rise_set["sunrise"].to_numpy()
    sunset = rise_set["sunset"].to_numpy()
    stamps = index.to_numpy()
    one_hour = np.timedelta64(1, "h")
    day_length = (sunset - sunrise) / one_hour
    hours_since_sunrise = (stamps - sunrise) / one_hour
    hours_to_sunset = (sunset - stamps) / one_hour

    is_daylight = (elevation > 0.0).astype("float64")

    return {
        "cos_zenith": cos_zenith.astype("float64"),
        "solar_elevation": elevation,
        "azimuth_sin": np.sin(azimuth),
        "azimuth_cos": np.cos(azimuth),
        "airmass": airmass,
        "extra_radiation": extra,
        "day_length": np.nan_to_num(day_length.astype("float64"), nan=0.0),
        "hours_since_sunrise": np.nan_to_num(hours_since_sunrise.astype("float64"), nan=0.0),
        "hours_to_sunset": np.nan_to_num(hours_to_sunset.astype("float64"), nan=0.0),
        "is_daylight": is_daylight,
    }
