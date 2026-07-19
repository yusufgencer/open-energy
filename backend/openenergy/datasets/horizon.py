from __future__ import annotations

import duckdb
import polars as pl

from openenergy.providers.base import ApiRole

# Canonical dtypes for well-known column names returned by weather/production queries
_CANONICAL_COL_DTYPES: dict[str, pl.PolarsDataType] = {
    "valid_time": pl.Datetime("us"),
    "issue_time": pl.Datetime("us"),
    "lead_hours": pl.Int32,
    "variable": pl.Utf8,
    "value": pl.Float64,
    "model": pl.Utf8,
    "power_mw": pl.Float64,
    "quality_flagged": pl.Boolean,
    "quality_severe": pl.Boolean,
}


def _fetchall_as_polars(result: duckdb.DuckDBPyRelation | duckdb.DuckDBPyConnection,
                        execute_result=None) -> pl.DataFrame:
    """Convert DuckDB query result to polars DataFrame without pyarrow.

    Preserves canonical dtypes (Datetime, Int32, Float64) for known column names
    so that empty and non-empty results have the same schema.
    """
    if execute_result is None:
        execute_result = result
    cols = [d[0] for d in execute_result.description]
    rows = execute_result.fetchall()
    if not rows:
        return pl.DataFrame(schema={c: _CANONICAL_COL_DTYPES.get(c, pl.Utf8) for c in cols})
    df = pl.DataFrame([dict(zip(cols, row)) for row in rows])
    cast_exprs = [
        pl.col(c).cast(_CANONICAL_COL_DTYPES[c])
        for c in cols
        if c in _CANONICAL_COL_DTYPES
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df


def build_horizon_weather(con: duckdb.DuckDBPyConnection, *, point_id: int,
                          horizon_hours: int, source: str | None = None,
                          variables: list[str] | None = None,
                          valid_time=None) -> pl.DataFrame:
    """Build the canonical previous-runs weather frame for one lead.

    This is the shared train/serve data-generating path: both consumers select
    ``role=previous_runs AND lead_hours=horizon_hours`` and pivot with identical
    semantics. Serving may additionally pin one ``valid_time``; training leaves
    it unset and receives the full historical row universe.

    With no explicit source, preserve the legacy single-source behaviour while
    refusing an ambiguous multi-model pivot.
    """
    params: list = [point_id, ApiRole.PREVIOUS_RUNS.value, horizon_hours]
    src_filter = ""
    if source:
        src_filter = "AND model=?"
        params.append(source)
    else:
        n_models = con.execute(
            "SELECT COUNT(DISTINCT model) FROM weather_raw "
            "WHERE point_id=? AND role=? AND lead_hours=?",
            [point_id, ApiRole.PREVIOUS_RUNS.value, horizon_hours],
        ).fetchone()[0]
        if n_models > 1:
            raise ValueError(
                f"belirsiz çok-kaynak: point_id={point_id}, lead={horizon_hours}h için "
                f"{n_models} farklı NWP model ingest edilmiş; 'source' açıkça verilmeli "
                f"(ambiguous multi-source — specify source=)"
            )

    var_filter = ""
    if variables:
        placeholders = ",".join(["?"] * len(variables))
        var_filter = f"AND variable IN ({placeholders})"
        params.extend(variables)

    valid_filter = ""
    if valid_time is not None:
        valid_filter = "AND valid_time=?"
        params.append(valid_time)

    result = con.execute(
        f"""
        SELECT valid_time, variable, value
        FROM weather_raw
        WHERE point_id=? AND role=? AND lead_hours=?
              {src_filter} {var_filter} {valid_filter}
        ORDER BY valid_time
        """,
        params,
    )
    long = _fetchall_as_polars(con, result)
    if long.height == 0:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us")})
    return (
        long.pivot(values="value", index="valid_time", on="variable",
                   aggregate_function="first")
        .sort("valid_time")
    )


def build_multipoint_horizon_weather(
    con: duckdb.DuckDBPyConnection, *, point_ids: list[int],
    horizon_hours: int, source_policy: str | None = None,
    source: str | None = None, variables: list[str] | None = None,
    valid_time=None,
) -> tuple[pl.DataFrame | None, set[int]]:
    """Canonical previous-runs weather frame for grid/source policies.

    Returns the merged frame and the set of points that contributed at least one
    row. The layout is the longstanding multipoint layout used by training.
    """
    if source_policy is None:
        source_policy = source
    sources: list[str | None] = source_policy.split("+") if source_policy else [None]
    combo = len(sources) > 1

    merged: pl.DataFrame | None = None
    present: set[int] = set()
    for pid in point_ids:
        for src in sources:
            params: list = [pid, ApiRole.PREVIOUS_RUNS.value, horizon_hours]
            src_filter = ""
            if src:
                src_filter = "AND model=?"
                params.append(src)
            var_filter = ""
            if variables:
                placeholders = ",".join(["?"] * len(variables))
                var_filter = f"AND variable IN ({placeholders})"
                params.extend(variables)
            valid_filter = ""
            if valid_time is not None:
                valid_filter = "AND valid_time=?"
                params.append(valid_time)
            res = con.execute(
                f"SELECT valid_time, variable, value FROM weather_raw "
                f"WHERE point_id=? AND role=? AND lead_hours=? "
                f"{src_filter} {var_filter} {valid_filter} ORDER BY valid_time",
                params,
            )
            long = _fetchall_as_polars(con, res)
            if long.height == 0:
                continue
            wide = long.pivot(values="value", index="valid_time", on="variable",
                              aggregate_function="first")
            present.add(pid)
            suffix = f"__p{pid}__{src}" if combo else f"__p{pid}"
            wide = wide.rename({c: f"{c}{suffix}" for c in wide.columns if c != "valid_time"})
            merged = wide if merged is None else merged.join(wide, on="valid_time", how="inner")
    if merged is not None:
        merged = merged.sort("valid_time")
    return merged, present


def build_horizon_dataset(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                          horizon_hours: int,
                          source: str | None = None,
                          variables: list[str] | None = None) -> pl.DataFrame:
    wide = build_horizon_weather(
        con, point_id=point_id, horizon_hours=horizon_hours,
        source=source, variables=variables,
    )
    if wide.height == 0:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us"), "power_mw": pl.Float64})

    prod_result = con.execute(
        """
        SELECT p.ts AS valid_time, p.power_mw,
               coalesce(q.quality_flagged, false) AS quality_flagged,
               coalesce(q.quality_severe, false) AS quality_severe
        FROM production p
        LEFT JOIN (
            SELECT plant_id, ts, true AS quality_flagged,
                   bool_or(severity = 'error') AS quality_severe
            FROM production_quality
            WHERE plant_id=?
            GROUP BY plant_id, ts
        ) q ON q.plant_id=p.plant_id AND q.ts=p.ts
        WHERE p.plant_id=?
        """,
        [plant_id, plant_id],
    )
    prod = _fetchall_as_polars(con, prod_result)

    if prod.height == 0:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us"), "power_mw": pl.Float64})

    result = wide.join(prod, on="valid_time", how="inner").sort("valid_time")
    return result


def build_pooled_dataset(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                         horizon_hours_list: list[int],
                         source: str | None = None,
                         variables: list[str] | None = None) -> pl.DataFrame:
    """Pool previous-runs weather across *several leads* into one long frame (F1-t2).

    Each lead's rows come from :func:`build_horizon_dataset` (the leak-safe
    ``role='previous_runs' AND lead_hours=lead`` path — every lead keeps its own
    weather values), tagged with a ``lead_time_hours`` column and stacked. The
    result is sorted by ``(valid_time, lead_time_hours)`` — a stable pooled row
    universe. Pooling introduces no cross-lead leakage by itself; downstream
    splits must partition by valid_time (see
    :func:`openenergy.evaluation.splits.group_time_split`) because the same
    ``valid_time`` now appears once per lead.
    """
    frames: list[pl.DataFrame] = []
    for lead in sorted(dict.fromkeys(horizon_hours_list)):
        f = build_horizon_dataset(con, plant_id=plant_id, point_id=point_id,
                                  horizon_hours=lead, source=source, variables=variables)
        if f.height == 0:
            continue
        frames.append(f.with_columns(pl.lit(lead).cast(pl.Int32).alias("lead_time_hours")))
    if not frames:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us"), "power_mw": pl.Float64,
                                    "lead_time_hours": pl.Int32})
    pooled = pl.concat(frames, how="diagonal")
    return pooled.sort(["valid_time", "lead_time_hours"])


def build_multipoint_dataset(con: duckdb.DuckDBPyConnection, *, plant_id: str,
                             point_ids: list[int], horizon_hours: int,
                             source_policy: str | None = None,
                             source: str | None = None,
                             variables: list[str] | None = None) -> pl.DataFrame:
    """Raw dataset over a grid of points, weather columns namespaced per point.

    Each point's variables become `{var}__p{point_id}` columns; points are inner-
    joined on valid_time (row universe = shared timestamps) and joined to production.
    Sorted by valid_time — the stable row universe over which point_policy then
    collapses columns.

    `source_policy` selects which NWP model(s) feed the frame (weather_raw.model):

    - ``None``            — no model filter (default / legacy, byte-identical).
    - ``"ecmwf"``         — single source: `AND model='ecmwf'`, columns stay
                            `{var}__p{pid}` (byte-identical layout to the single-
                            source path).
    - ``"ecmwf+icon_eu"`` — combo: each source is fetched separately and its
                            columns are further namespaced `{var}__p{pid}__{source}`
                            so the model sees BOTH sources side by side. The row
                            universe is the intersection of every (point, source).

    ``source`` is the legacy single-source alias (used when source_policy is None).
    """
    prod_result = con.execute(
        """
        SELECT p.ts AS valid_time, p.power_mw,
               coalesce(q.quality_flagged, false) AS quality_flagged,
               coalesce(q.quality_severe, false) AS quality_severe
        FROM production p
        LEFT JOIN (
            SELECT plant_id, ts, true AS quality_flagged,
                   bool_or(severity = 'error') AS quality_severe
            FROM production_quality
            WHERE plant_id=?
            GROUP BY plant_id, ts
        ) q ON q.plant_id=p.plant_id AND q.ts=p.ts
        WHERE p.plant_id=?
        """,
        [plant_id, plant_id],
    )
    prod = _fetchall_as_polars(con, prod_result)
    if prod.height == 0:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us"), "power_mw": pl.Float64})

    merged, _present = build_multipoint_horizon_weather(
        con, point_ids=point_ids, horizon_hours=horizon_hours,
        source_policy=source_policy, source=source, variables=variables,
    )
    if merged is None:
        return pl.DataFrame(schema={"valid_time": pl.Datetime("us"), "power_mw": pl.Float64})
    return merged.join(prod, on="valid_time", how="inner").sort("valid_time")
