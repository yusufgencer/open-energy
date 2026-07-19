"""EPİAŞ santral-bazlı üretim verisi — powerplant listesi + gerçek zamanlı üretim.

Fonksiyonlar `.get(path)` / `.post(path, body)` arayüzüne sahip bir client alır
(EpiasClient ya da test sahtesi) — böylece ağ/credential olmadan test edilebilir.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Protocol

import polars as pl


class _ClientLike(Protocol):
    def get(self, path: str) -> dict: ...
    def post(self, path: str, body: dict) -> dict: ...


def _epias_date(d: dt.date) -> str:
    # EPİAŞ tarih formatı: YYYY-MM-DDT00:00:00+03:00 (Türkiye saati)
    return f"{d.isoformat()}T00:00:00+03:00"


def _items(payload: dict) -> list[dict]:
    """Yanıt zarfından `items` listesini çıkar (birkaç olası şekil)."""
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("items"), list):
        return payload["items"]
    body = payload.get("body")
    if isinstance(body, dict):
        for v in body.values():
            if isinstance(v, list):
                return v
    return []


def list_powerplants(client: _ClientLike) -> list[dict]:
    """Gerçek zamanlı üretim santral listesi — her öğede id, name, eic."""
    return _items(client.get("/generation/data/powerplant-list"))


def find_powerplant(client: _ClientLike, name_substring: str) -> dict | None:
    """İsme göre (büyük/küçük harf duyarsız, kısmi) santral bul."""
    key = name_substring.upper()
    for it in list_powerplants(client):
        name = str(it.get("name") or it.get("shortName") or "").upper()
        if key in name:
            return it
    return None


def _parse_ts(raw: Any) -> dt.datetime | None:
    if raw is None:
        return None
    try:
        t = dt.datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t


def _validate_range(start: dt.date, end: dt.date) -> None:
    if start > end:
        raise ValueError(
            f"EPİAŞ tarih aralığı geçersiz: start={start.isoformat()} "
            f"end={end.isoformat()}"
        )


def _number(item: dict, fields: tuple[str, ...], *, endpoint: str, index: int) -> float:
    """Read one finite, non-negative MW value with endpoint-specific errors."""
    field = next((name for name in fields if name in item), None)
    if field is None:
        raise ValueError(
            f"EPİAŞ {endpoint} item {index}: beklenen MW alanı yok "
            f"({', '.join(fields)})"
        )
    raw = item[field]
    if raw is None:
        raise ValueError(f"EPİAŞ {endpoint} item {index}: '{field}' null olamaz")
    if isinstance(raw, bool):
        raise ValueError(f"EPİAŞ {endpoint} item {index}: '{field}' sayısal olmalı")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"EPİAŞ {endpoint} item {index}: '{field}' sayısal olmalı; "
            f"gelen={raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"EPİAŞ {endpoint} item {index}: '{field}' sonlu olmalı")
    if value < 0:
        raise ValueError(f"EPİAŞ {endpoint} item {index}: '{field}' negatif olamaz")
    return value


def _signal_frame(
    payload: dict,
    *,
    endpoint: str,
    value_column: str,
    value_fields: tuple[str, ...],
) -> pl.DataFrame:
    schema = {"ts": pl.Datetime("us"), value_column: pl.Float64}
    rows: list[tuple[dt.datetime, float]] = []
    items = _items(payload)
    has_list = isinstance(payload, dict) and (
        isinstance(payload.get("items"), list)
        or (
            isinstance(payload.get("body"), dict)
            and any(isinstance(value, list) for value in payload["body"].values())
        )
    )
    if not has_list:
        raise ValueError(
            f"EPİAŞ {endpoint} yanıtı beklenen 'items' listesini içermiyor"
        )
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"EPİAŞ {endpoint} item {index}: nesne olmalı")
        raw_ts = item.get("date") or item.get("dateTime") or item.get("hour")
        ts = _parse_ts(raw_ts)
        if ts is None:
            raise ValueError(
                f"EPİAŞ {endpoint} item {index}: geçerli bir tarih/saat yok; "
                f"gelen={raw_ts!r}"
            )
        rows.append((
            ts,
            _number(item, value_fields, endpoint=endpoint, index=index),
        ))
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, schema=["ts", value_column], orient="row")
        .with_columns(
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col(value_column).cast(pl.Float64),
        )
        .unique(subset=["ts"], keep="last")
        .sort("ts")
    )


def fetch_aic(
    client: _ClientLike,
    power_plant_id: int,
    start: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    """Fetch EPİAŞ emre amade kapasite (EAK/AIC).

    Returns a normalized ``(ts UTC-naive, aic_mw Float64)`` frame. EPİAŞ has
    used both English and Turkish value keys across response versions; aliases
    are accepted at this provider boundary, while callers see one stable schema.
    """
    _validate_range(start, end)
    payload = client.post(
        "/generation/data/aic",
        {
            "startDate": _epias_date(start),
            "endDate": _epias_date(end),
            "powerPlantId": int(power_plant_id),
        },
    )
    return _signal_frame(
        payload,
        endpoint="aic",
        value_column="aic_mw",
        value_fields=(
            "aic",
            "availableCapacity",
            "availableCapacityMw",
            "emreAmadeKapasite",
            "emreAmadeKapasiteMiktari",
            "emreAmadeKapasiteMiktarı",
            "total",
        ),
    )


def fetch_dpp(
    client: _ClientLike,
    power_plant_id: int,
    start: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    """Fetch EPİAŞ kesinleşmiş günlük üretim planı (KGÜP/DPP).

    Returns a normalized ``(ts UTC-naive, dpp_mw Float64)`` frame.
    """
    _validate_range(start, end)
    payload = client.post(
        "/generation/data/dpp",
        {
            "startDate": _epias_date(start),
            "endDate": _epias_date(end),
            "powerPlantId": int(power_plant_id),
        },
    )
    return _signal_frame(
        payload,
        endpoint="dpp",
        value_column="dpp_mw",
        value_fields=(
            "dpp",
            "dayAheadProductionPlan",
            "plannedProduction",
            "productionPlan",
            "kgup",
            "kgupMiktari",
            "kgupMiktarı",
            "total",
        ),
    )


def fetch_realtime_generation(
    client: _ClientLike,
    power_plant_id: int,
    start: dt.date,
    end: dt.date,
    *,
    field: str = "total",
) -> pl.DataFrame:
    """Santral-bazlı saatlik gerçek zamanlı üretim → (ts UTC-naive, power_mw).

    `field`: saf RES için "total" (=santralin toplam üretimi) ya da "wind".
    Dönen frame write_production ile uyumludur.
    """
    _validate_range(start, end)
    body = {
        "startDate": _epias_date(start),
        "endDate": _epias_date(end),
        "powerPlantId": power_plant_id,
    }
    payload = client.post("/generation/data/realtime-generation", body)
    rows: list[tuple[dt.datetime, float]] = []
    for it in _items(payload):
        ts = _parse_ts(it.get("date") or it.get("dateTime") or it.get("hour"))
        val = it.get(field)
        if ts is None or val is None:
            continue
        rows.append((ts, float(val)))
    schema = {"ts": pl.Datetime("us"), "power_mw": pl.Float64}
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, schema=["ts", "power_mw"], orient="row")
        .with_columns(pl.col("ts").cast(pl.Datetime("us")), pl.col("power_mw").cast(pl.Float64))
        .unique(subset=["ts"], keep="last")
        .sort("ts")
    )
