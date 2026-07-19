import datetime as dt

import httpx
import numpy as np
import respx

from openenergy.assets.repository import Plant, PlantPoint, add_point, create_plant
from openenergy.experiments.persistence import (
    create_experiment,
    mark_strategy_champion,
    promote_strategy_champion,
    record_strategy_candidate,
    record_strategy_trial,
)
from openenergy.experiments.search_space import PipelineConfig
from openenergy.experiments.strategy import FittedCandidate, StrategyArtifact, StrategyConfig
from openenergy.forecasting.predict import run_serving_cycle
from openenergy.models.linear_model import RidgeModel
from openenergy.providers.base import ApiRole
from openenergy.providers.openmeteo.client import BASE_URLS, OpenMeteoClient
from openenergy.providers.openmeteo.provider import OpenMeteoProvider


def _install_minimal_champion(db, tmp_path) -> int:
    """Persist a real strategy artifact and promote it through the registry."""
    pipeline = PipelineConfig(
        model_family="ridge",
        nwp_source="best_match",
        feature_blocks=[],
        params={"alpha": 1.0},
    )
    model = RidgeModel(alpha=1.0)
    model.fit(
        np.array([[4.0], [8.0]], dtype=float),
        np.array([0.2, 0.6], dtype=float),
    )

    model_path = tmp_path / "serving-e2e-ridge.pkl"
    model.save(model_path)
    candidate = FittedCandidate(
        config=pipeline,
        model=model,
        feature_blocks=[],
        feature_names=["wind_speed_100m"],
        cv_nrmse=0.1,
    )
    artifact_path = tmp_path / "serving-e2e-strategy.pkl"
    StrategyArtifact(candidates=[candidate], ensemble_method="none").save(artifact_path)

    experiment_id = create_experiment(db, "wf-e2e", [24])
    strategy_config = StrategyConfig(
        candidates=[pipeline],
        ensemble_method="none",
        top_k=1,
    )
    strategy_trial_id = record_strategy_trial(
        db,
        experiment_id=experiment_id,
        horizon_hours=24,
        strategy_config=strategy_config,
        cv_nrmse=0.1,
        test_metrics={"nrmse": 0.1},
        skill_score=0.2,
        artifact_path=str(artifact_path),
    )
    record_strategy_candidate(
        db,
        strategy_trial_id=strategy_trial_id,
        rank=1,
        candidate_config=pipeline,
        cv_nrmse=0.1,
        test_nrmse=0.1,
        artifact_path=str(model_path),
        feature_names=["wind_speed_100m"],
    )
    mark_strategy_champion(db, experiment_id, 24, strategy_trial_id)
    promote_strategy_champion(db, "wf-e2e", 24, strategy_trial_id)
    return strategy_trial_id


@respx.mock
def test_forecast_served_end_to_end_from_provider(db, tmp_path):
    """HTTP previous-runs -> parser -> weather_raw -> champion -> forecasts."""
    create_plant(
        db,
        Plant(
            plant_id="wf-e2e",
            name="Serving E2E",
            kind="wind",
            capacity_mw=10.0,
        ),
    )
    point = add_point(
        db,
        PlantPoint(
            plant_id="wf-e2e",
            point_type="weather",
            latitude=39.9,
            longitude=32.8,
            hub_height_m=100.0,
        ),
    )
    strategy_trial_id = _install_minimal_champion(db, tmp_path)

    issue_time = dt.datetime(2024, 6, 1)
    target_time = issue_time + dt.timedelta(hours=24)
    payload = {
        "latitude": 39.9,
        "longitude": 32.8,
        "timezone": "GMT",
        "hourly": {
            "time": [target_time.isoformat(timespec="minutes")],
            "wind_speed_100m_previous_day1": [8.5],
            "wind_speed_10m_previous_day1": [5.1],
            "temperature_2m_previous_day1": [18.0],
            "surface_pressure_previous_day1": [1013.0],
        },
    }
    route = respx.get(BASE_URLS[ApiRole.PREVIOUS_RUNS]).mock(
        return_value=httpx.Response(200, json=payload)
    )
    provider = OpenMeteoProvider(OpenMeteoClient(max_retries=0))

    result = run_serving_cycle(
        db,
        plant_id="wf-e2e",
        point=point,
        horizons=[24],
        capacity_mw=10.0,
        kind="wind",
        issue_time=issue_time,
        provider=provider,
    )

    assert route.call_count == 1
    assert result["ingest"]["rows"] == 4
    assert db.execute(
        """
        SELECT count(*) FROM weather_raw
        WHERE point_id=? AND role='previous_runs' AND lead_hours=24
          AND valid_time=?
        """,
        [point.point_id, target_time],
    ).fetchone()[0] == 4

    forecasts = db.execute(
        """
        SELECT issue_time, valid_time, p50, strategy_trial_id
        FROM forecasts
        WHERE plant_id='wf-e2e'
        """
    ).fetchall()
    assert result["rows"] >= 1
    assert len(forecasts) >= 1
    assert forecasts[0][0] == issue_time
    assert forecasts[0][1] == target_time
    assert 0.0 <= forecasts[0][2] <= 10.0
    assert forecasts[0][3] == strategy_trial_id
