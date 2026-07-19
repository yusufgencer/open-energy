"""EPİAŞ üretim ve operatör sinyali ingestion'ı."""
from __future__ import annotations

import datetime as dt

import duckdb

from openenergy.ingestion.production import write_production
from openenergy.providers.epias.client import EpiasClient
from openenergy.providers.epias.generation import (
    fetch_aic,
    fetch_dpp,
    fetch_realtime_generation,
    find_powerplant,
)


def _resolve_powerplant(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    *,
    power_plant_id: int | None,
    plant_name: str | None,
    client,
) -> tuple[int, str | None]:
    """Resolve and persist the internal plant ↔ EPİAŞ plant mapping."""
    if con.execute(
        "SELECT 1 FROM plants WHERE plant_id=?", [plant_id]
    ).fetchone() is None:
        raise ValueError(f"Bilinmeyen plant_id: {plant_id}")

    if power_plant_id is None:
        saved = con.execute(
            "SELECT power_plant_id, power_plant_name "
            "FROM epias_plant_mapping WHERE plant_id=?",
            [plant_id],
        ).fetchone()
        if saved is not None:
            return int(saved[0]), saved[1]

        lookup_name = plant_name
        if not lookup_name:
            lookup_name = con.execute(
                "SELECT name FROM plants WHERE plant_id=?", [plant_id]
            ).fetchone()[0]
        match = find_powerplant(client, lookup_name)
        if match is None:
            raise ValueError(f"EPİAŞ'ta '{lookup_name}' adında santral bulunamadı")
        try:
            power_plant_id = int(match["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"EPİAŞ santral eşleşmesinde geçerli 'id' yok: {match!r}"
            ) from exc
        resolved_name = str(match.get("name") or match.get("shortName") or "") or None
    else:
        try:
            power_plant_id = int(power_plant_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("power_plant_id tam sayı olmalı") from exc
        if power_plant_id <= 0:
            raise ValueError("power_plant_id pozitif olmalı")
        resolved_name = plant_name

    con.execute(
        "INSERT INTO epias_plant_mapping "
        "(plant_id, power_plant_id, power_plant_name, updated_at) "
        "VALUES (?, ?, ?, now()) "
        "ON CONFLICT (plant_id) DO UPDATE SET "
        "power_plant_id=excluded.power_plant_id, "
        "power_plant_name=excluded.power_plant_name, updated_at=now()",
        [plant_id, power_plant_id, resolved_name],
    )
    return power_plant_id, resolved_name


def _write_signal(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    frame,
    *,
    table: str,
    value_column: str,
) -> int:
    """Atomically and idempotently replace one normalized EPİAŞ signal frame."""
    if table not in {"epias_aic", "epias_dpp"}:
        raise ValueError(f"Desteklenmeyen EPİAŞ sinyal tablosu: {table}")
    if frame.height == 0:
        return 0
    rows = (
        frame.select("ts", value_column)
        .unique(subset=["ts"], keep="last", maintain_order=True)
        .rows()
    )
    placeholders = ", ".join(["(?, ?, ?, now())"] * len(rows))
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            f"INSERT OR REPLACE INTO {table} "
            f"(plant_id, ts, {value_column}, ingested_at) VALUES {placeholders}",
            [value for ts, mw in rows for value in (plant_id, ts, mw)],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def ingest_epias_aic(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    *,
    start: dt.date,
    end: dt.date,
    power_plant_id: int | None = None,
    plant_name: str | None = None,
    client=None,
) -> dict:
    """Fetch and idempotently store EPİAŞ EAK/AIC availability."""
    client = client or EpiasClient()
    external_id, resolved_name = _resolve_powerplant(
        con,
        plant_id,
        power_plant_id=power_plant_id,
        plant_name=plant_name,
        client=client,
    )
    frame = fetch_aic(client, external_id, start, end)
    rows = _write_signal(
        con, plant_id, frame, table="epias_aic", value_column="aic_mw"
    )
    return {
        "rows": rows,
        "power_plant_id": external_id,
        "resolved_name": resolved_name,
        "ts_min": str(frame["ts"].min()) if frame.height else None,
        "ts_max": str(frame["ts"].max()) if frame.height else None,
    }


def ingest_epias_dpp(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    *,
    start: dt.date,
    end: dt.date,
    power_plant_id: int | None = None,
    plant_name: str | None = None,
    client=None,
) -> dict:
    """Fetch and idempotently store EPİAŞ KGÜP/DPP production plans."""
    client = client or EpiasClient()
    external_id, resolved_name = _resolve_powerplant(
        con,
        plant_id,
        power_plant_id=power_plant_id,
        plant_name=plant_name,
        client=client,
    )
    frame = fetch_dpp(client, external_id, start, end)
    rows = _write_signal(
        con, plant_id, frame, table="epias_dpp", value_column="dpp_mw"
    )
    return {
        "rows": rows,
        "power_plant_id": external_id,
        "resolved_name": resolved_name,
        "ts_min": str(frame["ts"].min()) if frame.height else None,
        "ts_max": str(frame["ts"].max()) if frame.height else None,
    }


def ingest_epias_production(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    *,
    start: dt.date,
    end: dt.date,
    power_plant_id: int | None = None,
    plant_name: str | None = None,
    field: str = "total",
    client=None,
) -> dict:
    """EPİAŞ'tan santral üretimini çek ve `production`'a yaz.

    `power_plant_id` verilmezse `plant_name` ile powerplant-list üzerinden çözülür.
    Returns: {"rows", "power_plant_id", "resolved_name", "ts_min", "ts_max"}
    """
    client = client or EpiasClient()

    power_plant_id, resolved_name = _resolve_powerplant(
        con,
        plant_id,
        power_plant_id=power_plant_id,
        plant_name=plant_name,
        client=client,
    )

    frame = fetch_realtime_generation(client, power_plant_id, start, end, field=field)
    rows = write_production(
        con,
        plant_id,
        frame,
        source=f"epias:realtime-generation:{field}",
        revision=f"{start.isoformat()}_{end.isoformat()}",
    )
    return {
        "rows": rows,
        "power_plant_id": power_plant_id,
        "resolved_name": resolved_name,
        "ts_min": str(frame["ts"].min()) if frame.height else None,
        "ts_max": str(frame["ts"].max()) if frame.height else None,
    }
