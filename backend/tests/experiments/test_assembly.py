import datetime as dt
import numpy as np
import polars as pl
from openenergy.assets.repository import Plant, create_plant
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.features import FeatureConfig
from openenergy.experiments.assembly import assemble, featurize
from openenergy.datasets.horizon import build_multipoint_dataset


def _prev_series(point_id, n=50, model="best_match", offset=0.0):
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("wind_speed_100m", 5 + (i % 7) + offset), ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
            rows["valid_time"].append(vt); rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24); rows["variable"].append(var); rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    return WeatherSeries(point_id=point_id, role=ApiRole.PREVIOUS_RUNS.value, model=model, frame=frame), vts


def test_assemble_excludes_target_from_features(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    series, vts = _prev_series(7)
    write_weather_series(db, series)
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(2 + (i % 5)) for i in range(len(vts))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "wf1", prod)
    cfg = FeatureConfig(blocks=["wind_power", "air_density"])
    ds = assemble(db, plant_id="wf1", point_id=7, horizon_hours=24, capacity_mw=10.0, kind="wind", feature_config=cfg)
    assert ds.X.shape[0] == len(vts)
    assert "power_mw" not in ds.feature_names and "power_norm" not in ds.feature_names and "valid_time" not in ds.feature_names
    assert "wind_speed_cube" in ds.feature_names
    assert ds.y.max() <= 1.2 and ds.y.min() >= 0.0


def test_multipoint_featurize_point_policies_stable_rows(db):
    create_plant(db, Plant(plant_id="mp", name="MP", kind="wind", capacity_mw=10))
    for pid in (7, 8):
        series, vts = _prev_series(pid)
        write_weather_series(db, series)
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(2 + (i % 5)) for i in range(len(vts))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "mp", prod)

    raw = build_multipoint_dataset(db, plant_id="mp", point_ids=[7, 8], horizon_hours=24)
    assert any(c.endswith("__p7") for c in raw.columns)  # namespaced per point
    assert raw.height == len(vts)

    cfg = FeatureConfig(blocks=["wind_power"])
    ds_mean = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0,
                        point_policy="spatial_mean", point_ids=[7, 8])
    ds_spread = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0,
                          point_policy="mean_plus_spread", point_ids=[7, 8])
    ds_all = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0,
                       point_policy="all_points", point_ids=[7, 8])
    # rows identical across policies; columns differ (the axis is real)
    assert ds_mean.X.shape[0] == ds_spread.X.shape[0] == ds_all.X.shape[0] == raw.height
    assert ds_all.X.shape[1] > ds_mean.X.shape[1]  # all_points keeps per-point columns
    assert ds_spread.X.shape[1] > ds_mean.X.shape[1]  # spread adds *_sstd


def _write_source(db, plant_id, point_id, model, offset, n=50):
    series, vts = _prev_series(point_id, n=n, model=model, offset=offset)
    write_weather_series(db, series)
    return vts


def test_source_policy_single_source_sees_only_its_rows(db):
    create_plant(db, Plant(plant_id="sp", name="SP", kind="wind", capacity_mw=10))
    vts = _write_source(db, "sp", 7, "ecmwf", offset=0.0)
    _write_source(db, "sp", 7, "icon_eu", offset=100.0)
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(2 + (i % 5)) for i in range(len(vts))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "sp", prod)

    ec = build_multipoint_dataset(db, plant_id="sp", point_ids=[7], horizon_hours=24,
                                  source_policy="ecmwf")
    ic = build_multipoint_dataset(db, plant_id="sp", point_ids=[7], horizon_hours=24,
                                  source_policy="icon_eu")
    # single-source columns are NOT source-namespaced (byte-identical layout to today)
    assert "wind_speed_100m__p7" in ec.columns
    assert not any(c.endswith("__ecmwf") for c in ec.columns)
    # each source sees only its own values (ecmwf base vs icon_eu +100 offset)
    assert ec["wind_speed_100m__p7"].max() < 50
    assert ic["wind_speed_100m__p7"].min() >= 100
    assert ec.height == ic.height == len(vts)


def test_source_policy_combo_sees_both_namespaces(db):
    create_plant(db, Plant(plant_id="sp2", name="SP2", kind="wind", capacity_mw=10))
    vts = _write_source(db, "sp2", 7, "ecmwf", offset=0.0)
    _write_source(db, "sp2", 7, "icon_eu", offset=100.0)
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(2 + (i % 5)) for i in range(len(vts))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "sp2", prod)

    combo = build_multipoint_dataset(db, plant_id="sp2", point_ids=[7], horizon_hours=24,
                                     source_policy="ecmwf+icon_eu")
    # combo yields per-source namespaced columns so the model sees BOTH
    assert "wind_speed_100m__p7__ecmwf" in combo.columns
    assert "wind_speed_100m__p7__icon_eu" in combo.columns
    assert combo["wind_speed_100m__p7__ecmwf"].max() < 50
    assert combo["wind_speed_100m__p7__icon_eu"].min() >= 100


def test_source_policy_combo_row_universe_is_intersection(db):
    create_plant(db, Plant(plant_id="sp3", name="SP3", kind="wind", capacity_mw=10))
    vts_ec = _write_source(db, "sp3", 7, "ecmwf", offset=0.0, n=50)
    _write_source(db, "sp3", 7, "icon_eu", offset=100.0, n=40)  # fewer timestamps
    prod = pl.DataFrame({"ts": vts_ec, "power_mw": [float(2 + (i % 5)) for i in range(len(vts_ec))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "sp3", prod)

    combo = build_multipoint_dataset(db, plant_id="sp3", point_ids=[7], horizon_hours=24,
                                     source_policy="ecmwf+icon_eu")
    # row universe = intersection of the two sources' timestamps (40, not 50)
    assert combo.height == 40


# --- kPV target policy (C1-t3) ---

def _clear_sky_solar_raw(capacity_mw=10.0, frac=0.75):
    """Synthetic clear-sky-shaped production spanning winter→summer.

    Production is a fixed fraction of the clear-sky envelope P_cs. Winter rows
    (lower sun) come first, summer rows (higher peaks) last — so a train-on-early
    / test-on-late split forces the model to predict peaks it never saw at
    training time (the extrapolation trap capacity_norm falls into and kpv escapes).
    Returns (raw_frame, geometry, p_cs_array).
    """
    from openenergy.features.target import PlantGeometry
    from openenergy.physics.solar import clear_sky_power

    # Near-horizontal tilt so summer clear-sky peaks genuinely exceed winter.
    geom = PlantGeometry(lat=39.9, lon=32.8, tilt=5.0, azimuth=180.0)
    times = []
    # 20 winter days (earlier in time) + 20 summer days, hourly — sorted so the
    # low-peak winter block trains and the high-peak summer block tests.
    for base in (dt.datetime(2023, 12, 1), dt.datetime(2024, 6, 10)):
        for d in range(20):
            for h in range(24):
                times.append(base + dt.timedelta(days=d, hours=h))
    times.sort()
    p_cs = clear_sky_power(times, geom.lat, geom.lon, geom.tilt, geom.azimuth,
                           capacity_mw).to_numpy()
    power = frac * p_cs
    # shortwave-like feature: a clean proxy of the envelope the model can read
    shortwave = 1000.0 * p_cs / capacity_mw
    raw = pl.DataFrame(
        {
            "valid_time": times,
            "shortwave_radiation": shortwave.astype(float),
            "power_mw": power.astype(float),
        },
        schema_overrides={"valid_time": pl.Datetime("us")},
    )
    return raw, geom, power


def test_featurize_kpv_carries_ratio_target_and_pcs():
    raw, geom, power = _clear_sky_solar_raw()
    ds = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                   capacity_mw=10.0, target_policy="kpv", geometry=geom)
    assert ds.target_policy == "kpv"
    assert ds.p_cs is not None and ds.p_cs.shape[0] == raw.height
    # daylight target rows cluster near the fixed clear-sky ratio (0.75)
    day = ds.p_cs > 1e-6
    np.testing.assert_allclose(ds.y[day], 0.75, rtol=1e-6)
    # night rows: zero target
    assert float(np.max(ds.y[~day])) == 0.0


def test_featurize_capacity_norm_unchanged_by_kpv_addition():
    raw, geom, power = _clear_sky_solar_raw()
    ds = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                   capacity_mw=10.0)
    assert ds.target_policy == "capacity_norm"
    assert ds.p_cs is None
    np.testing.assert_allclose(ds.y, power / 10.0, rtol=1e-9)


# --- daylight masking (C1-t4) ---


def test_featurize_daylight_mask_drops_night_rows_for_solar():
    from openenergy.physics.solar import daylight_mask as _dmask

    raw, geom, power = _clear_sky_solar_raw()
    full = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                     capacity_mw=10.0, geometry=geom)
    masked = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                       capacity_mw=10.0, geometry=geom, daylight_mask=True)
    # some rows are night → strictly fewer training rows after masking
    expected_keep = int(_dmask(raw["valid_time"].to_list(), geom.lat, geom.lon).sum())
    assert 0 < masked.X.shape[0] == expected_keep < full.X.shape[0]
    # X, y and valid_time stay row-aligned after the drop
    assert masked.y.shape[0] == masked.X.shape[0] == masked.valid_time.shape[0]
    assert masked.X.shape[1] == full.X.shape[1]  # columns unchanged


def test_featurize_daylight_mask_kept_rows_are_daytime():
    raw, geom, power = _clear_sky_solar_raw()
    masked = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                       capacity_mw=10.0, target_policy="kpv", geometry=geom, daylight_mask=True)
    # kpv envelope survives the mask row-aligned and every kept row is daytime (P_cs>0)
    assert masked.p_cs is not None and masked.p_cs.shape[0] == masked.X.shape[0]
    assert float(masked.p_cs.min()) > 0.0


def test_featurize_daylight_mask_off_keeps_all_rows():
    raw, geom, power = _clear_sky_solar_raw()
    ds = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                   capacity_mw=10.0, geometry=geom)  # daylight_mask defaults off
    assert ds.X.shape[0] == raw.height


def test_featurize_daylight_mask_ignored_for_wind(db):
    # Wind unaffected: masking flag is a no-op for wind (no night rows dropped).
    create_plant(db, Plant(plant_id="wfd", name="WF", kind="wind", capacity_mw=10))
    series, vts = _prev_series(7)
    write_weather_series(db, series)
    prod = pl.DataFrame({"ts": vts, "power_mw": [float(2 + (i % 5)) for i in range(len(vts))]},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "wfd", prod)
    from openenergy.experiments.assembly import assemble_raw
    raw = assemble_raw(db, plant_id="wfd", point_id=7, horizon_hours=24)
    cfg = FeatureConfig(blocks=["wind_power"])
    ds = featurize(raw, kind="wind", feature_config=cfg, capacity_mw=10.0, daylight_mask=True)
    assert ds.X.shape[0] == raw.height  # every row kept


def test_kpv_reproduces_peak_that_capacity_norm_underestimates():
    from openenergy.features.target import kpv_inverse
    from openenergy.models.lightgbm_model import LightGBMModel

    cap = 10.0
    raw, geom, power = _clear_sky_solar_raw(capacity_mw=cap, frac=0.75)
    n = raw.height
    # train on the early (winter) half, test on the late (summer) half
    tr = np.arange(n // 2)
    te = np.arange(n // 2, n)

    params = {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1, "verbose": -1}

    ds_cap = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                       capacity_mw=cap)
    m_cap = LightGBMModel(params=params)
    m_cap.fit(ds_cap.X[tr], ds_cap.y[tr])
    pred_cap_mw = np.clip(m_cap.predict(ds_cap.X[te]).p50 * cap, 0, None)

    ds_kpv = featurize(raw, kind="solar", feature_config=FeatureConfig(blocks=["cyclical_time"]),
                       capacity_mw=cap, target_policy="kpv", geometry=geom)
    m_kpv = LightGBMModel(params=params)
    m_kpv.fit(ds_kpv.X[tr], ds_kpv.y[tr])
    pred_kpv_mw = kpv_inverse(m_kpv.predict(ds_kpv.X[te]).p50, ds_kpv.p_cs[te], cap)

    actual_peak = float(np.max(power[te]))
    cap_peak = float(np.max(pred_cap_mw))
    kpv_peak = float(np.max(pred_kpv_mw))

    # capacity_norm cannot extrapolate to the higher summer peak → underestimates
    assert cap_peak < 0.9 * actual_peak
    # kpv reconstructs the summer peak from physics
    assert kpv_peak > 0.9 * actual_peak
    assert kpv_peak <= cap  # never above nameplate
    assert kpv_peak > cap_peak
