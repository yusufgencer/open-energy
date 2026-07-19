import datetime as dt

import polars as pl

from openenergy.assets.repository import Plant, create_plant
from openenergy.datasets.horizon import build_horizon_dataset
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production


def _prev_runs_series(point_id):
    # iki valid_time, iki lead bucket (24 ve 48), tek değişken
    rows = []
    for vt in [dt.datetime(2024, 6, 2, 0), dt.datetime(2024, 6, 2, 1)]:
        for lead, val in [(24, 8.5), (48, 7.0)]:
            rows.append((vt, vt - dt.timedelta(hours=lead), lead, "wind_speed_100m", val))
    frame = pl.DataFrame(
        {
            "valid_time": [r[0] for r in rows],
            "issue_time": [r[1] for r in rows],
            "lead_hours": [r[2] for r in rows],
            "variable": [r[3] for r in rows],
            "value": [r[4] for r in rows],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    return WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                         model="best_match", frame=frame)


def _setup(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    write_weather_series(db, _prev_runs_series(point_id=7))
    prod = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 6, 2, 0), dt.datetime(2024, 6, 2, 1)],
            "power_mw": [3.0, 3.5],
        },
        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
    )
    write_production(db, "wf1", prod)


def test_dataset_uses_only_matching_lead(db):
    _setup(db)
    ds = build_horizon_dataset(db, plant_id="wf1", point_id=7, horizon_hours=24)
    # 2 valid_time satırı, wind_speed_100m == 8.5 (24h bucket), 48h bucket karışmamalı
    assert ds.height == 2
    assert "wind_speed_100m" in ds.columns
    assert ds.sort("valid_time")["wind_speed_100m"][0] == 8.5
    assert ds.sort("valid_time")["power_mw"][0] == 3.0


def test_dataset_different_horizon_picks_different_value(db):
    _setup(db)
    ds = build_horizon_dataset(db, plant_id="wf1", point_id=7, horizon_hours=48)
    assert ds.sort("valid_time")["wind_speed_100m"][0] == 7.0


def test_row_universe_invariant_under_flags(db):
    _setup(db)
    before = build_horizon_dataset(
        db, plant_id="wf1", point_id=7, horizon_hours=24
    )
    flagged_ts = dt.datetime(2024, 6, 2, 1)
    db.execute(
        """
        INSERT INTO production_quality
          (plant_id, ts, flag_code, severity, details, source, revision,
           observed_power_mw)
        VALUES ('wf1', ?, 'NEGATIVE_POWER', 'error', '{}', 'test', 'r1', 3.5)
        """,
        [flagged_ts],
    )
    after = build_horizon_dataset(
        db, plant_id="wf1", point_id=7, horizon_hours=24
    )
    assert after["valid_time"].to_list() == before["valid_time"].to_list()
    assert after["power_mw"].to_list() == before["power_mw"].to_list()
    assert after.height == before.height == 2
    assert after["quality_severe"].to_list() == [False, True]


def test_dataset_no_weather_early_exit(db):
    """When no weather exists at the given lead bucket, returns empty with correct schema."""
    _setup(db)
    # horizon_hours=96 → no weather rows at that lead → early exit before join
    ds = build_horizon_dataset(db, plant_id="wf1", point_id=7, horizon_hours=96)
    assert ds.height == 0


def test_dataset_drops_rows_without_production(db):
    """Weather exists at lead=24 but production has no matching valid_time → inner-join drops all rows."""
    create_plant(db, Plant(plant_id="wf_noprod", name="WF NoProd", kind="wind", capacity_mw=10))
    # weather at 2024-06-02 00:00 and 01:00 with lead=24
    rows = [
        (dt.datetime(2024, 6, 2, 0), dt.datetime(2024, 6, 1, 0), 24, "wind_speed_100m", 8.5),
        (dt.datetime(2024, 6, 2, 1), dt.datetime(2024, 6, 1, 1), 24, "wind_speed_100m", 8.9),
    ]
    frame = pl.DataFrame(
        {
            "valid_time": [r[0] for r in rows],
            "issue_time": [r[1] for r in rows],
            "lead_hours": [r[2] for r in rows],
            "variable": [r[3] for r in rows],
            "value": [r[4] for r in rows],
        },
        schema_overrides={
            "valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32, "value": pl.Float64,
        },
    )
    series = WeatherSeries(point_id=99, role=ApiRole.PREVIOUS_RUNS.value,
                           model="best_match", frame=frame)
    write_weather_series(db, series)
    # production at a DIFFERENT valid_time → inner join produces 0 rows
    prod = pl.DataFrame(
        {"ts": [dt.datetime(2024, 6, 3, 0)], "power_mw": [3.0]},
        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
    )
    write_production(db, "wf_noprod", prod)
    ds = build_horizon_dataset(db, plant_id="wf_noprod", point_id=99, horizon_hours=24)
    assert ds.height == 0


# --- Finding 5: re-ingesting the same previous_runs series must not double row count,
#     and build_horizon_dataset must return the expected single value per (valid_time, variable) ---

def test_reingest_previous_runs_no_duplicates_in_dataset(db):
    _setup(db)
    series = _prev_runs_series(point_id=7)
    # Write the same data a second time
    write_weather_series(db, series)
    count = db.execute(
        "SELECT count(*) FROM weather_raw WHERE point_id=7"
    ).fetchone()[0]
    # 4 original rows (2 valid_times × 2 lead buckets) — must not double
    assert count == series.frame.height

    ds = build_horizon_dataset(db, plant_id="wf1", point_id=7, horizon_hours=24)
    # Still exactly 2 rows, single value per (valid_time, variable)
    assert ds.height == 2
    assert ds.sort("valid_time")["wind_speed_100m"][0] == 8.5


def _one_source(point_id, model, val, *, vt=dt.datetime(2024, 6, 2, 0), lead=24):
    frame = pl.DataFrame(
        {"valid_time": [vt], "issue_time": [vt - dt.timedelta(hours=lead)],
         "lead_hours": [lead], "variable": ["wind_speed_100m"], "value": [val]},
        schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                          "lead_hours": pl.Int32, "value": pl.Float64},
    )
    return WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value,
                         model=model, frame=frame)


def _setup_multisource(db):
    create_plant(db, Plant(plant_id="wfm", name="WFM", kind="wind", capacity_mw=10))
    write_weather_series(db, _one_source(7, "ecmwf", 8.5))
    write_weather_series(db, _one_source(7, "icon_eu", 6.0))
    write_production(db, "wfm", pl.DataFrame(
        {"ts": [dt.datetime(2024, 6, 2, 0)], "power_mw": [3.0]},
        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64}))


def test_build_horizon_dataset_raises_on_ambiguous_multi_source(db):
    """T-04: with two NWP models ingested for the same point/lead, source=None is
    ambiguous — the pivot would silently pick an arbitrary model. Must raise so a
    run's executed frame is a deterministic function of its recorded source."""
    import pytest
    _setup_multisource(db)
    with pytest.raises(ValueError, match="(?i)source|kaynak"):
        build_horizon_dataset(db, plant_id="wfm", point_id=7, horizon_hours=24, source=None)


def test_build_horizon_dataset_explicit_source_is_deterministic(db):
    """T-04: naming the source disambiguates — the frame carries exactly that
    model's value, no matter how many models are ingested."""
    _setup_multisource(db)
    ds = build_horizon_dataset(db, plant_id="wfm", point_id=7, horizon_hours=24, source="ecmwf")
    assert ds.height == 1
    assert ds["wind_speed_100m"][0] == 8.5


# --- F1-t2: pooled-over-leads dataset (one frame across leads, lead_time_hours column) ---

def test_build_pooled_dataset_stacks_leads_with_lead_time_hours_column():
    from openenergy.datasets.horizon import build_pooled_dataset
    import datetime as dt

    con = None  # placeholder to keep import order clear
    from openenergy.storage import connect, init_schema
    con = connect(None)
    init_schema(con)
    try:
        create_plant(con, Plant(plant_id="wfp", name="WFP", kind="wind", capacity_mw=10))
        # weather at leads 24 AND 48 for the same two valid_times (distinct values)
        con_series = _prev_runs_series(point_id=7)
        write_weather_series(con, con_series)  # leads 24 (8.5) and 48 (7.0)
        prod = pl.DataFrame(
            {"ts": [dt.datetime(2024, 6, 2, 0), dt.datetime(2024, 6, 2, 1)],
             "power_mw": [3.0, 3.5]},
            schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
        )
        write_production(con, "wfp", prod)

        pooled = build_pooled_dataset(con, plant_id="wfp", point_id=7,
                                      horizon_hours_list=[24, 48])
        # the pooled frame tags every row with its lead
        assert "lead_time_hours" in pooled.columns
        assert set(pooled["lead_time_hours"].unique().to_list()) == {24, 48}
        # rows pooled across both leads: 2 valid_times × 2 leads
        assert pooled.height == 4
        # each lead keeps its own weather value (24→8.5, 48→7.0), no mixing
        v24 = pooled.filter(pl.col("lead_time_hours") == 24)["wind_speed_100m"].to_list()
        v48 = pooled.filter(pl.col("lead_time_hours") == 48)["wind_speed_100m"].to_list()
        assert set(v24) == {8.5}
        assert set(v48) == {7.0}
    finally:
        con.close()
