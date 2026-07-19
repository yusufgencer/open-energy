import datetime as dt

import numpy as np
import polars as pl

from openenergy.assets.repository import Plant, create_plant
from openenergy.datasets.horizon import build_horizon_dataset
from openenergy.experiments.assembly import featurize
from openenergy.features import FeatureConfig
from openenergy.forecasting.assembly import forecast_matrix
from openenergy.ingestion.production import write_production
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.providers.base import ApiRole, WeatherSeries


def test_serving_features_identical_to_training_features(db):
    """Same point, valid time and lead must produce identical train/serve X."""
    point_id = 13
    plant_id = "skew"
    horizon = 24
    issue_time = dt.datetime(2026, 7, 17, 0, 0)
    valid_time = issue_time + dt.timedelta(hours=horizon)
    create_plant(
        db, Plant(plant_id=plant_id, name="Skew", kind="wind", capacity_mw=20),
    )
    rows = {
        "valid_time": [],
        "issue_time": [],
        "lead_hours": [],
        "variable": [],
        "value": [],
    }
    values = {
        "wind_speed_100m": 8.25,
        "wind_speed_10m": 5.5,
        "temperature_2m": 17.0,
        "surface_pressure": 1007.0,
    }
    for variable, value in values.items():
        rows["valid_time"].append(valid_time)
        rows["issue_time"].append(issue_time)
        rows["lead_hours"].append(horizon)
        rows["variable"].append(variable)
        rows["value"].append(value)
    frame = pl.DataFrame(
        rows,
        schema_overrides={
            "valid_time": pl.Datetime("us"),
            "issue_time": pl.Datetime("us"),
            "lead_hours": pl.Int32,
            "value": pl.Float64,
        },
    )
    write_weather_series(
        db,
        WeatherSeries(
            point_id=point_id,
            role=ApiRole.PREVIOUS_RUNS.value,
            model="best_match",
            frame=frame,
        ),
    )
    write_production(
        db,
        plant_id,
        pl.DataFrame(
            {"ts": [valid_time], "power_mw": [6.0]},
            schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64},
        ),
    )

    config = FeatureConfig(blocks=["wind_power", "air_density"])
    raw = build_horizon_dataset(
        db, plant_id=plant_id, point_id=point_id, horizon_hours=horizon,
    )
    training = featurize(
        raw, kind="wind", feature_config=config, capacity_mw=20,
    )
    serving_x, serving_vt = forecast_matrix(
        db,
        point_id=point_id,
        kind="wind",
        feature_blocks=config.blocks,
        feature_names=training.feature_names,
        issue_time=issue_time,
        horizon_hours=horizon,
    )

    assert training.feature_names
    assert serving_vt.tolist() == training.valid_time.tolist()
    assert serving_x.shape == training.X.shape
    assert np.array_equal(serving_x, training.X)
